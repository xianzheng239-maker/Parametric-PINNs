# -*- coding: utf-8 -*-
"""
myPINN_target_control - 稳定版（整合防护与诊断）

说明：
- 在原有“可微分一致 + 分步更新”实现的基础上，加入了数值稳定性保护：
  * min_s, _small_eps 的提升与使用
  * lambda clamp（上下界）
  * B_diag 强制下限与上限保护
  * 更保守的 s 步长（step_scale_s，可调）
  * 全局梯度裁剪
  * 训练前/后数值检查 (NaN/Inf 检测)
  * debug_print 开关：初期建议打开以观察 s, lambda, ffun 分布

- 目标：保持你原来的分步更新逻辑（对角 BFGS、投影、lambda = mu/s），同时将
  可能出现 NaN/Inf 的点用防护措施钳制，降低训练出现数值错误的概率。

使用：替换你原来的类，直接调用 model.compile(...) 和 model.fit(dataset, ...)

作者: ChatGPT
"""

import tensorflow as tf
import numpy as np

# metrics
loss_tracker = tf.keras.metrics.Mean(name="loss")
loss_tracker_u = tf.keras.metrics.Mean(name="lossu")
loss_tracker_f_all = tf.keras.metrics.Mean(name="lossf_all")
loss_tracker_f = tf.keras.metrics.Mean(name="lossf")
loss_tracker_n = tf.keras.metrics.Mean(name="lossn")
loss_tracker_c = tf.keras.metrics.Mean(name="lossc")
loss_tracker_f_add1 = tf.keras.metrics.Mean(name="lossf_add1")
loss_tracker_f_add2 = tf.keras.metrics.Mean(name="lossf_add2")

mae_metric_u = tf.keras.metrics.MeanAbsoluteError(name="maeu")

tf.keras.backend.set_floatx('float64')


class myPINN_target_control(tf.keras.Model):
    """可微分一致 + 分步更新（含数值稳健保护）

    主要改动（相对于你提供的实现）:
      - 增强数值保护: min_s, _small_eps, lambda 上下界, B_diag 下限/上限
      - 梯度裁剪 (clip_by_global_norm)
      - 更保守的 s 更新步长 step_scale_s
      - 训练过程中 NaN/Inf 检测和诊断打印 (可通过 debug_print 控制)

    接口与原类一致，直接替换并运行。
    """

    def __init__(self, dataSet, parameters, myPDE, myPDF, net_pdf, net_control, mu_init=1e-3):
        super().__init__()

        # references
        self.dataSet = dataSet
        self.parameters = parameters
        self.myPDE = myPDE
        self.myPDF = myPDF
        self.net_pdf = net_pdf
        self.net_control = net_control

        # system params
        self.C_par = tf.constant(self.parameters.system_par["C_par"], dtype=tf.float64)
        self.R_par = tf.constant(self.parameters.system_par["R_par"], dtype=tf.float64)

        # boundary data
        self.par_b = tf.constant(dataSet.IBCdata.par_BIC_arr, dtype=tf.float64)
        self.x_b = tf.constant(dataSet.IBCdata.x_BIC_arr, dtype=tf.float64)
        self.x_data_b = [self.par_b, self.x_b]
        self.y_b = tf.constant(dataSet.IBCdata.u_BIC_arr, dtype=tf.float64)

        # ML weights
        self.weight_uf = parameters.ml_parameter["weight_uf"]
        self.weight_f = parameters.ml_parameter["weight_f"]
        self.weight_l1 = parameters.ml_parameter.get("weight_l1", 0.0)
        self.weight_l2 = parameters.ml_parameter.get("weight_l2", 0.0)
        self.weight_u = parameters.ml_parameter.get("weight_u", 1.0)
        self.weight_n = parameters.ml_parameter.get("weight_n", 1.0)
        self.weight_c = parameters.ml_parameter.get("weight_c", 1.0)

        # barrier and numerics
        self.mu = tf.Variable(mu_init, dtype=tf.float64, trainable=False)
        self.epsilon = tf.constant(1e-2, dtype=tf.float64)
        # slightly larger epsilon for numerical stability
        self._small_eps = tf.constant(1e-8, dtype=tf.float64)
        self.min_s = tf.constant(1e-4, dtype=tf.float64)       # lower bound for s
        self.max_lambda = tf.constant(1e6, dtype=tf.float64)   # clamp lambda
        self.max_B = tf.constant(1e12, dtype=tf.float64)       # clamp B_diag

        # s update step scale & grad clipping
        self.step_scale_s = tf.constant(0.1, dtype=tf.float64)  # conservative default
        self.grad_clip_norm = 5.0

        # training points
        self.Num_training = parameters.axis_par["Num_training"]
        self.x_f_tf = tf.constant(dataSet.Trainingdata.x_f_arr, dtype=tf.float64)
        self.par_f_tf = tf.constant(dataSet.Trainingdata.par_f_arr, dtype=tf.float64)
        self.x_f_data = [self.par_f_tf, self.x_f_tf]

        # init s shape
        ffun = self.myPDE.FPK_loss(self.net_pdf, self.net_control, self.x_f_data)
        ffun = tf.convert_to_tensor(ffun, dtype=tf.float64)

        init_s = tf.ones_like(ffun, dtype=tf.float64) * 1.0
        init_s = tf.maximum(init_s, self.min_s)
        self.s = tf.Variable(init_s, dtype=tf.float64, trainable=False)

        # init lambda
        init_lambda = tf.ones_like(ffun, dtype=tf.float64) * (float(mu_init) / (1.0 + float(self._small_eps)))
        init_lambda = tf.clip_by_value(init_lambda, 0.0, float(self.max_lambda.numpy()))
        self.lambda_var = tf.Variable(init_lambda, dtype=tf.float64, trainable=False)

        # diagonal BFGS approx
        self.B_diag = tf.Variable(tf.ones_like(self.s, dtype=tf.float64), dtype=tf.float64, trainable=False)
        self.prev_gradient = tf.Variable(tf.zeros_like(self.s, dtype=tf.float64), dtype=tf.float64, trainable=False)
        self.prev_s = tf.Variable(tf.zeros_like(self.s, dtype=tf.float64), dtype=tf.float64, trainable=False)
        self.first_update = True

        # debug
        self.debug_print = False

    @property
    def metrics(self):
        return [loss_tracker, loss_tracker_u, loss_tracker_f_all, loss_tracker_n, loss_tracker_c]

    def lagrangian(self, x_data, use_barrier=True):
        """构造 Lagrangian。s 与 lambda_var 使用当前 class 值（在计算图内可见）。"""
        mylossu = self.loss_data(self.x_data_b, self.y_b)
        mylossf = self.loss_residual(x_data)
        mylossn = self.loss_normalization_GL()
        mylossc = self.loss_cost_GL()

        main_loss = self.weight_uf * (
            self.weight_u * mylossu + self.weight_f * mylossf + self.weight_n * mylossn + self.weight_c * mylossc
        )

        # PDE residual
        ffun = self.myPDE.FPK_loss(self.net_pdf, self.net_control, x_data)
        ffun = tf.convert_to_tensor(ffun, dtype=tf.float64)
        ffun = tf.reshape(ffun, tf.shape(self.s))

        constraint_term = tf.reduce_sum(self.lambda_var * (ffun + self.s - self.epsilon))

        barrier_term = tf.constant(0.0, dtype=tf.float64)
        if use_barrier:
            # use log(max(s, min_s)) for numerical safety
            safe_s = tf.maximum(self.s, self.min_s)
            barrier_term = - self.mu * tf.reduce_sum(tf.math.log(safe_s))

        total = main_loss + constraint_term + barrier_term
        return total, mylossu, mylossf, mylossf

    def train_step(self, data):
        x, y = data

        # ------------------ Stage 1: update network params using Lagrangian ------------------
        with tf.GradientTape() as tape:
            L_total, mylossu, mylossf, mylossf = self.lagrangian(self.x_f_data, use_barrier=True)

            # diagnostic: check L_total
            nan_or_inf = tf.logical_or(
                tf.reduce_any(tf.math.is_nan(L_total)),
                tf.reduce_any(tf.math.is_inf(L_total))
            )
            tf.cond(
                nan_or_inf,
                lambda: tf.print("⚠️ NaN or Inf detected:", L_total),
                lambda: tf.no_op()
            )

        # exclude s, lambda, mu, B_diag, prev_* from network update
        excluded = {id(self.s), id(self.lambda_var), id(self.mu), id(self.B_diag), id(self.prev_gradient), id(self.prev_s)}
        net_vars = [v for v in self.trainable_variables if id(v) not in excluded]

        grads = tape.gradient(L_total, net_vars)
        grads = [tf.zeros_like(v) if g is None else g for v, g in zip(net_vars, grads)]

        # gradient clipping for stability
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)

        # check grads numerics
        for g in grads:
            nan_or_inf_g = tf.logical_or(
                tf.reduce_any(tf.math.is_nan(g)),
                tf.reduce_any(tf.math.is_inf(g))
            )
            tf.cond(
                nan_or_inf_g,
                lambda: tf.print("Found NaN/Inf in gradients. Inspect grad stats."),
                lambda: tf.no_op()
            )

        self.optimizer.apply_gradients(zip(grads, net_vars))

        # ------------------ Stage 2: update s (BFGS-like) and lambda, mu ------------------
        with tf.GradientTape() as tape_s:
            tape_s.watch(self.s)
            L_for_s, _, _, _ = self.lagrangian(self.x_f_data, use_barrier=True)

        dL_ds = tape_s.gradient(L_for_s, self.s)
        if dL_ds is None:
            if self.debug_print:
                tf.print("Warning: dL/ds is None")
        else:
            self._update_s_by_diag_bfgs(dL_ds)

        # lambda = mu / s but clamp s to min_s to avoid huge lambda
        safe_s = tf.maximum(self.s, self.min_s)
        new_lambda = tf.math.divide_no_nan(self.mu, safe_s)
        new_lambda = tf.clip_by_value(new_lambda, 0.0, self.max_lambda)
        self.lambda_var.assign(new_lambda)

        # update mu schedule
        self.update_mu(self.x_f_data)

        # ------------------ metrics & diagnostics ------------------
        # recompute main_loss components for metrics (use standard loss not L_total)
        mylossu = self.loss_data(x, y)
        mylossf = self.loss_residual(x)
        mylossn = self.loss_normalization_GL()
        mylossc = self.loss_cost_GL()

        main_loss = self.weight_uf * (
            self.weight_u * mylossu + self.weight_f * mylossf + self.weight_n * mylossn + self.weight_c * mylossc
        )

        loss_tracker.update_state(main_loss)
        loss_tracker_u.update_state(mylossu)
        loss_tracker_f.update_state(mylossf)
        loss_tracker_n.update_state(mylossn)
        loss_tracker_c.update_state(mylossc)

        if self.debug_print:
            tf.print("train_step diag:",
                     "mean s", tf.reduce_mean(self.s), "min s", tf.reduce_min(self.s), "max s", tf.reduce_max(self.s),
                     "mean lambda", tf.reduce_mean(self.lambda_var), "mu", self.mu,
                     "main_loss", main_loss)

        barrier_term = self.mu * tf.reduce_sum(tf.math.log(tf.maximum(self.s, self.min_s)))

        return {"barrier_term": barrier_term, "loss": loss_tracker.result(), "lossu": loss_tracker_u.result(),
                "lossf": loss_tracker_f.result(),
                "lossn": loss_tracker_n.result(), "lossc": loss_tracker_c.result()}

    def _update_s_by_diag_bfgs(self, dL_ds):
        """对角 BFGS 风格更新 + 投影，增加了数值保护（clamp B_diag、使用 step_scale_s）。"""
        dL_ds_vec = tf.reshape(dL_ds, [-1, 1])
        s_vec = tf.reshape(self.s, [-1, 1])

        # 保证 B_diag 在合理范围
        self.B_diag.assign(tf.clip_by_value(self.B_diag, 1e-8, float(self.max_B.numpy())))
        B_diag_vec = tf.reshape(self.B_diag, [-1, 1])

        if self.first_update:
            p_s = - dL_ds_vec / (B_diag_vec + self._small_eps)
        else:
            y = dL_ds_vec - tf.reshape(self.prev_gradient, [-1, 1])
            s_diff = s_vec - tf.reshape(self.prev_s, [-1, 1])
            ys_dot = tf.reduce_sum(y * s_diff)

            def _update_B():
                y_flat = tf.reshape(y, [-1])
                B_flat = tf.reshape(self.B_diag, [-1])
                sdiff_flat = tf.reshape(s_diff, [-1])
                denom = ys_dot + self._small_eps
                new_B_flat = B_flat - (B_flat * sdiff_flat) ** 2 / denom + (y_flat ** 2) / denom
                new_B_flat = tf.clip_by_value(new_B_flat, 1e-8, float(self.max_B.numpy()))
                self.B_diag.assign(tf.reshape(new_B_flat, tf.shape(self.B_diag)))
                return True

            tf.cond(tf.abs(ys_dot) > 1e-12, _update_B, lambda: False)
            B_diag_vec = tf.reshape(self.B_diag, [-1, 1])
            p_s = - dL_ds_vec / (B_diag_vec + self._small_eps)

        # conservative step scale
        step_scale = self.step_scale_s
        p_s = tf.reshape(p_s, tf.shape(s_vec))
        p_s = tf.clip_by_value(p_s, -1e6, 1e6)
        new_s = s_vec + step_scale * p_s

        # projection: ensure ffun + s >= epsilon and s >= min_s
        ffun = self.myPDE.FPK_loss(self.net_pdf, self.net_control, self.x_f_data)
        ffun = tf.convert_to_tensor(ffun, dtype=tf.float64)
        ffun_vec = tf.reshape(ffun, [-1, 1])

        lower_bound = -ffun_vec + self.epsilon
        lower_bound = tf.maximum(lower_bound, self.min_s)
        new_s = tf.maximum(new_s, lower_bound)
        new_s = tf.maximum(new_s, self.min_s)

        new_s_assign = tf.reshape(new_s, tf.shape(self.s))
        self.s.assign(new_s_assign)

        # 更新历史量
        self.prev_gradient.assign(tf.reshape(dL_ds_vec, tf.shape(self.prev_gradient)))
        self.prev_s.assign(tf.reshape(s_vec, tf.shape(self.prev_s)))
        if self.first_update:
            self.first_update = False

        if self.debug_print:
            tf.print("update_s: mean(s)", tf.reduce_mean(self.s), "mean|dL/ds|", tf.reduce_mean(tf.abs(dL_ds)))

    def update_mu(self, x_data):
        """简单的 mu 衰减策略：当最大残差小于阈值时加速衰减。"""
        ffun = self.myPDE.FPK_loss(self.net_pdf, self.net_control, x_data)
        ffun = tf.convert_to_tensor(ffun, dtype=tf.float64)
        violation = tf.reduce_max(tf.abs(ffun))
        #decay_factor = tf.cond(violation < tf.constant(1e-4, dtype=tf.float64), lambda: tf.constant(0.87, dtype=tf.float64), lambda: tf.constant(0.97, dtype=tf.float64))
        decay_factor = tf.cond(violation < tf.constant(1e-4, dtype=tf.float64),
                               lambda: tf.constant(0.8, dtype=tf.float64), lambda: tf.constant(0.95, dtype=tf.float64))

        new_mu = self.mu * decay_factor
        new_mu = tf.maximum(new_mu, tf.constant(1e-4, dtype=tf.float64))
        self.mu.assign(new_mu)

    #    @tf.function
    def loss_all(self, x_data):
        mylossu = self.loss_data(self.x_data_b, self.y_b)
        mylossf = self.loss_residual(x_data)
        mylossn = self.loss_normalization_GL()
        mylossc = self.loss_cost_GL()
        loss = self.weight_uf * (self.weight_u * mylossu + self.weight_f*mylossf + self.weight_n * mylossn + self.weight_c * mylossc)

        barrier_term = self.mu * tf.math.log(self.s)
        loss = loss - barrier_term

        return barrier_term,loss, mylossu, mylossf,mylossn, mylossc

    # 以下保留原始损失函数接口
    def loss_data(self, x_data_b, y_b):
        y_pred = self.net_pdf(x_data_b, training=True)
        mylossu = tf.math.reduce_mean(tf.math.square(y_pred - y_b))
        return mylossu

    def loss_residual(self, x_data):
        ffun = self.myPDE.FPK_loss(self.net_pdf, self.net_control, x_data)
        ffun = tf.convert_to_tensor(ffun, dtype=tf.float64)
        mylossf = tf.math.reduce_mean(tf.math.square(ffun))


        return  mylossf

    def loss_normalization_GL(self):
        y_pred_N_arr = self.net_pdf([self.dataSet.Normalizingdata.Par_N_mat2arr, self.dataSet.Normalizingdata.Xt_auxi_gl_mat2arr], training=True)
        y_pred_N = tf.reshape(y_pred_N_arr, self.dataSet.Normalizingdata.Par_t1_N_mat.shape)
        int_y_GL = tf.matmul(self.dataSet.Normalizingdata.weight_auxi_gl_mat2arr, y_pred_N)
        mylossn = tf.math.reduce_mean(tf.math.square(int_y_GL - 1.0))
        return mylossn

    def loss_cost_GL(self):
        pdf_pred_N_arr = self.net_pdf([self.dataSet.Costdata.Par_N_mat2arr, self.dataSet.Costdata.Xt_auxi_gl_mat2arr], training=True)
        control_pred_N_arr = self.net_control([self.dataSet.Costdata.Par_N_mat2arr, self.dataSet.Costdata.Xt_auxi_gl_mat2arr], training=True)
        part1_arr = (pdf_pred_N_arr- self.dataSet.Costdata.targetPDF) ** 2
        part1_mat = tf.reshape(part1_arr, self.dataSet.Costdata.Par_t1_N_mat.shape)
        int1 = tf.matmul(self.dataSet.Costdata.weight_auxi_gl_mat2arr, part1_mat)

        part2_arr = tf.reduce_sum(control_pred_N_arr, axis=1, keepdims=True) ** 2 * pdf_pred_N_arr
        part2_mat = tf.reshape(part2_arr, self.dataSet.Costdata.Par_t1_N_mat.shape)
        int2 = tf.matmul(self.dataSet.Costdata.weight_auxi_gl_mat2arr, part2_mat)
        mycost = tf.math.reduce_mean(self.C_par * int1 + self.R_par * int2)
        return mycost

# Usage notes:
# - 将此文件保存为 myPINN_target_control_stable_v1.py 并在你的项目中 import。
# - 推荐配置：optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)；开启 debug_print = True 在前几步观察。
# - 如需更强保护，可把 step_scale_s 降到 0.005 或更小。

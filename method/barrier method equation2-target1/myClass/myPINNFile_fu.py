# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:24:24 2020

@author: WantaoJia
"""
import math

import tensorflow as tf
import numpy as np
from scipy.optimize import minimize
import scipy
import parameters

loss_tracker = tf.keras.metrics.Mean(name="loss")
loss_tracker_u = tf.keras.metrics.Mean(name="lossu")
loss_tracker_f_all = tf.keras.metrics.Mean(name="lossf_all")
loss_tracker_f = tf.keras.metrics.Mean(name="lossf")
loss_tracker_f_add1 = tf.keras.metrics.Mean(name="lossf_add1")
loss_tracker_f_add2 = tf.keras.metrics.Mean(name="lossf_add2")
loss_tracker_n = tf.keras.metrics.Mean(name="lossn")
loss_tracker_c = tf.keras.metrics.Mean(name="lossc")

mae_metric_u = tf.keras.metrics.MeanAbsoluteError(name="maeu")
mae_metric_f = tf.keras.metrics.MeanAbsoluteError(name="maef")

tf.keras.backend.set_floatx('float64')


class myPINN_target_control(tf.keras.Model):
    def __init__(self, dataSet, parameters, myPDE, myPDF, net_pdf, net_control):
        super().__init__()
        ###############################################################################
        self.parameters = parameters
        self.dataSet = dataSet
        ###############################################################################
        self.C_par = tf.constant(self.parameters.system_par["C_par"], dtype=tf.float64)
        self.R_par = tf.constant(self.parameters.system_par["R_par"], dtype=tf.float64)
        ###############################################################################
        self.par_b = tf.constant(dataSet.IBCdata.par_BIC_arr, dtype=tf.float64)
        self.x_b = tf.constant(dataSet.IBCdata.x_BIC_arr, dtype=tf.float64)
        self.x_data_b = [self.par_b, self.x_b]
        self.y_b = tf.constant(dataSet.IBCdata.u_BIC_arr, dtype=tf.float64)
        ###############################################################################
        self.myPDE = myPDE
        self.myPDF = myPDF
        self.net_pdf = net_pdf
        self.net_control = net_control
        ###############################################################################
        self.parameters = parameters
        self.weight_f = parameters.ml_parameter["weight_f"]
        self.weight_l1 = parameters.ml_parameter["weight_l1"]
        self.weight_l2 = parameters.ml_parameter["weight_l2"]
        self.weight_u = parameters.ml_parameter["weight_u"]
        self.weight_uf = parameters.ml_parameter["weight_uf"]
        self.weight_n = parameters.ml_parameter["weight_n"]
        self.weight_c = parameters.ml_parameter["weight_c"]

        self.mu = tf.Variable(1.0, dtype=tf.float64)
        self.decay_rate = 0.7
        self.learning_rate = tf.constant(0.0001, dtype=tf.float64)

        # 添加松弛变量初始化
        self.epsilon = 1e-3  # 小的正数
        self.delta = 1e-5  # 另一个小的正数

        self.Num_training = parameters.axis_par["Num_training"]
        self.x_f_tf = tf.constant(dataSet.Trainingdata.x_f_arr, dtype=tf.float64)
        self.par_f_tf = tf.constant(dataSet.Trainingdata.par_f_arr, dtype=tf.float64)

        self.x_f_data = [self.par_f_tf, self.x_f_tf]



        # 计算初始残差
        add_loss1, add_loss2, ffun = self.myPDE.yf_detailed_balance(
            self.net_pdf, self.net_control, self.x_f_data)
        initial_r = tf.math.reduce_mean(tf.math.square(ffun))

        # 初始化松弛变量（符合文档公式17-18）
        initial_s = tf.math.maximum(self.epsilon-initial_r, self.delta)

        # 使用矩阵存储多个松弛变量（文档要求多个松弛变量）
        self.s = tf.Variable(
            tf.ones_like(ffun, dtype=tf.float64) * initial_s,  # 每个点有自己的松弛变量
            dtype=tf.float64,
            trainable=True
        )
        print(ffun.shape)

        # 拉格朗日乘子初始化
        self.lambda_var = tf.Variable(
            tf.zeros_like(ffun, dtype=tf.float64),
            dtype=tf.float64,
            trainable=True
        )

        # BFGS矩阵近似初始化
        self.B = tf.Variable(tf.eye(ffun.shape[-1], dtype=tf.float64))

    ###############################################################################
    @tf.function
    def train_step(self, data):
        # data transfered
        x, y = data

        # 更新松弛变量和拉格朗日乘子
        self.update_slack_variables(x)
        self.update_lagrange_multipliers(x)


        """
        with tf.GradientTape() as tape:
            loss = self.lagrangian(x)  # 使用拉格朗日函数作为损失

        # 更新网络参数
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        """

        # 更新障碍参数
        self.update_mu(x)

        with tf.GradientTape() as tape:


            # 临时替换为常量版本
            original_s = self.s
            original_lambda = self.lambda_var
            original_mu=self.mu


            barrier_term,loss, mylossu, mylossf_all, mylossf, mylossf_add1, mylossf_add2, mylossn, mylossc = self.loss_all(x)

            # 恢复原始变量
            self.s = original_s
            self.lambda_var = original_lambda
            self.mu=original_mu

        # 只更新网络参数（不包括s和lambda和mu）
        net_vars = [var for var in self.trainable_variables
                    if var is not self.s and var is not self.lambda_var and var is not self.mu]
        gradients = tape.gradient(loss, net_vars)
        self.optimizer.apply_gradients(zip(gradients, net_vars))

        # trainable_vars = self.trainable_variables
        # gradients = tape.gradient(loss, trainable_vars)
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        loss_tracker.update_state(loss)
        loss_tracker_u.update_state(mylossu)
        loss_tracker_f_all.update_state(mylossf_all)
        loss_tracker_f.update_state(mylossf)
        loss_tracker_f_add1.update_state(mylossf_add1)
        loss_tracker_f_add2.update_state(mylossf_add2)
        loss_tracker_n.update_state(mylossn)
        loss_tracker_c.update_state(mylossc)


        return {"barrier_term":barrier_term,"loss": loss_tracker.result(), "lossu": loss_tracker_u.result(),
                "lossf_all": loss_tracker_f_all.result(),
                "lossf": loss_tracker_f.result(), "lossf_add1": loss_tracker_f_add1.result(),
                "lossf_add2": loss_tracker_f_add2.result(),
                "lossn": loss_tracker_n.result(), "lossc": loss_tracker_c.result()}

    @property
    def metrics(self):
        # We list our "Metric" objects here so that 'reset_stats()' can be called
        # automatically at the start of each epoch

        return [loss_tracker, loss_tracker_u, loss_tracker_f, loss_tracker_n, loss_tracker_c]

    def update_slack_variables(self, x_data):
        """使用信任域SQP方法更新松弛变量（文档公式20-25）"""

        # 确保所有变量都被跟踪（文档公式19）
        with tf.GradientTape(persistent=True) as tape:
            # 显式watch Variable（文档公式20-22要求）
            tape.watch(self.s)
            tape.watch(self.lambda_var)

            # 计算拉格朗日函数（文档公式19）
            L = self.lagrangian(x_data)

            # 重新计算残差（确保使用最新的网络参数）
            _, _, ffun = self.myPDE.yf_detailed_balance(
                self.net_pdf, self.net_control, x_data)

        # 计算梯度（KKT条件，文档公式20-22）
        try:
            dL_ds = tape.gradient(L, self.s)
        except Exception as e:
            print(f"梯度计算错误: {e}")
            dL_ds = tf.zeros_like(self.s)

        # 处理梯度异常情况
        if dL_ds is None:
            dL_ds = tf.zeros_like(self.s)
        elif isinstance(dL_ds, tf.IndexedSlices):
            dL_ds = tf.convert_to_tensor(dL_ds)

        # 求解二次近似问题（文档公式23-24）
        p_s = -dL_ds  # 更新方向

        # 确保p_s是密集张量
        if isinstance(p_s, tf.IndexedSlices):
            p_s = tf.convert_to_tensor(p_s)

        # 更新松弛变量（文档公式25）

        new_s = self.s +  p_s

        # 应用约束：s >= epsilon（文档公式15-16）
        new_s = tf.maximum(new_s, self.epsilon)

        # 应用约束：s >= -ffun + epsilon（文档公式15-16）
        new_s = tf.maximum(new_s, -ffun + self.epsilon)

        # 确保使用assign方法更新Variable（关键修复）

        self.s.assign(new_s)


        return p_s

    def lagrangian(self, x_data):
        """计算拉格朗日函数（文档公式19）"""
        # 计算各项损失
        mylossu = self.loss_data(self.x_data_b, self.y_b)
        mylossf_all, mylossf, mylossf_add1, mylossf_add2 = self.loss_residual(x_data)
        mylossn = self.loss_normalization_GL()
        mylossc = self.loss_cost_GL()

        # 主损失项（对应文档中的f(x)）
        main_loss = self.weight_uf * (
                self.weight_u * mylossu +
                # self.weight_f * mylossf_all +
                self.weight_n * mylossn +
                self.weight_c * mylossc
        )


        # 计算残差（约束违反）
        _, _, ffun = self.myPDE.yf_detailed_balance(
            self.net_pdf, self.net_control, x_data)

        # 确保松弛变量和乘子已初始化
        if not hasattr(self, 's') or not hasattr(self, 'lambda_var'):
            # 如果未初始化，跳过障碍项和约束项
            return main_loss


        # 确保形状匹配 - 使用广播机制
        s = self.s
        lambda_var = self.lambda_var

        # 障碍项：μ * Σ(1/s)（文档公式19）
        barrier_term =  self.mu * tf.math.log(self.s)

        # 约束项：λ*(r) = λ*(ffun + s)（文档公式19）
        constraint_term = tf.reduce_sum(lambda_var * (ffun + s))

        # 总拉格朗日函数（文档公式19）
        total_lagrangian = main_loss + barrier_term + constraint_term

        return total_lagrangian

    def update_mu(self, x_data):
        """更新障碍参数μ（文档公式26-28）"""
        # 计算当前约束违反量
        _, _, ffun = self.myPDE.yf_detailed_balance(
            self.net_pdf, self.net_control, x_data)

        # 确保松弛变量已初始化
        if not hasattr(self, 's'):
            return
        s = self.s


        # 计算最大约束违反量（文档公式27）
        violation = tf.reduce_max(tf.abs(ffun + s))

        # 动态调整μ（文档公式27-28）
        if violation < 1e-4:
            # 衰减障碍参数
            new_mu_value = self.mu * 0.1  # 大幅衰减（文档公式27）

            # 确保最小值
            min_mu = tf.constant(1e-10, dtype=self.mu.dtype)
            new_mu_value = tf.maximum(new_mu_value, min_mu)
        else:
            # 衰减障碍参数
            new_mu_value = self.mu * 0.7  # 大幅衰减（文档公式27）

            # 确保最小值
            min_mu = tf.constant(1e-10, dtype=self.mu.dtype)
            new_mu_value = tf.maximum(new_mu_value, min_mu)



        # 确保μ不会太小（使用 TensorFlow 操作）
        self.mu.assign(tf.maximum(new_mu_value, 1e-10))  # 正确的方式！

    def bfgs_update(self, B, s, y):
        """简化BFGS更新（对角近似）"""
        s = tf.reshape(s, [-1])
        y = tf.reshape(y, [-1])

        # 计算对角更新
        rho = 1.0 / tf.reduce_sum(y * s)
        diag_B = tf.linalg.diag_part(B)

        # BFGS对角更新公式
        new_diag = diag_B - rho * (diag_B * s) ** 2 + rho * y ** 2

        return tf.linalg.diag(new_diag)

    def update_lagrange_multipliers(self, x_data):
        """更新拉格朗日乘子（文档公式21）"""
        # 计算当前残差
        _, _, ffun = self.myPDE.yf_detailed_balance(
            self.net_pdf, self.net_control, x_data)
        r = ffun + self.s

        # 更新乘子
        learning_rate = 1
        new_lambda = self.lambda_var + learning_rate * r
        if not hasattr(self, 'lambda_var') or not isinstance(self.lambda_var, tf.Variable):
            # 如果s不存在或不是Variable，创建新的Variable
            self.lambda_var = tf.Variable(new_lambda, dtype=tf.float64, trainable=True)
        else:
            # 使用assign更新现有Variable
            self.lambda_var.assign(new_lambda)

        # self.lambda_var.assign(new_lambda)



    #    @tf.function
    def loss_all(self, x_data):
        mylossu = self.loss_data(self.x_data_b, self.y_b)
        mylossf_all, mylossf, mylossf_add1, mylossf_add2 = self.loss_residual(x_data)
        mylossn = self.loss_normalization_GL()
        mylossc = self.loss_cost_GL()
        loss = self.weight_uf * (self.weight_u * mylossu + self.weight_n * mylossn + self.weight_c * mylossc)

        barrier_term = self.mu * tf.math.log(self.s)
        loss = loss - barrier_term

        return barrier_term,loss, mylossu, mylossf_all, mylossf, mylossf_add1, mylossf_add2, mylossn, mylossc

    #    @tf.function
    def loss_data(self, x_data_b, y_b):
        y_pred = self.net_pdf(x_data_b, training=True)
        mylossu = tf.math.reduce_mean(tf.math.square(y_pred - y_b))

        return mylossu

    #    @tf.function
    def loss_residual(self, x_data):
        add_loss1, add_loss2, ffun = self.myPDE.yf_detailed_balance(self.net_pdf, self.net_control, x_data)
        mylossf = tf.math.reduce_mean(tf.math.square(ffun))
        mylossf_add1 = tf.math.reduce_mean(tf.math.square(add_loss1))
        mylossf_add2 = tf.math.reduce_mean(tf.math.square(add_loss2))

        #mylossf_all = mylossf + self.weight_l1 * mylossf_add1 + self.weight_l2 * mylossf_add2
        mylossf_all =mylossf +self.weight_l1 * mylossf_add1 + self.weight_l2 * mylossf_add2

        return mylossf_all, mylossf, mylossf_add1, mylossf_add2

    #############################################################################################################
    #    @tf.function
    def loss_normalization_GL(self):

        y_pred_N_arr = self.net_pdf([self.dataSet.Normalizingdata.Par_N_mat2arr,
                                     self.dataSet.Normalizingdata.Xt_auxi_gl_mat2arr], training=True)
        y_pred_N = tf.reshape(y_pred_N_arr, self.dataSet.Normalizingdata.Par_t1_N_mat.shape)

        int_y_GL = tf.matmul(self.dataSet.Normalizingdata.weight_auxi_gl_mat2arr, y_pred_N)

        mylossn = tf.math.reduce_mean(tf.math.square(int_y_GL - 1.0))

        return mylossn

    #############################################################################################################
    #    @tf.function
    def loss_cost_GL(self):
        pdf_pred_N_arr = self.net_pdf([self.dataSet.Costdata.Par_N_mat2arr,
                                       self.dataSet.Costdata.Xt_auxi_gl_mat2arr], training=True)
        control_pred_N_arr = self.net_control([self.dataSet.Costdata.Par_N_mat2arr,
                                               self.dataSet.Costdata.Xt_auxi_gl_mat2arr], training=True)

        part1_arr = (pdf_pred_N_arr - self.dataSet.Costdata.targetPDF) ** 2
        part1_mat = tf.reshape(part1_arr, self.dataSet.Costdata.Par_t1_N_mat.shape)
        int1 = tf.matmul(self.dataSet.Costdata.weight_auxi_gl_mat2arr, part1_mat)

        part2_arr = tf.reduce_sum(control_pred_N_arr, axis=1, keepdims=True) ** 2 * pdf_pred_N_arr
        part2_mat = tf.reshape(part2_arr, self.dataSet.Costdata.Par_t1_N_mat.shape)
        int2 = tf.matmul(self.dataSet.Costdata.weight_auxi_gl_mat2arr, part2_mat)

        mycost = tf.math.reduce_mean(self.C_par * int1 + self.R_par * int2)

        return mycost

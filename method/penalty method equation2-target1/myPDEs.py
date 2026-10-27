# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:38:26 2023

@author: WantaoJia
"""

import numpy as np
import tensorflow as tf



class myPoissonFPKSDOF():
    def __init__(self,parameters,dataSet):
###############################################################################
        self.alpha1 = tf.constant(parameters.system_par["alpha1"],dtype=tf.float64)
        #self.beta1 = tf.constant(parameters.system_par["beta1"],dtype=tf.float64)
        self.D11 = tf.constant(parameters.noise_par["D11"],dtype=tf.float64)
        self.S11 = tf.constant(parameters.noise_par["S11"],dtype=tf.float64)


        self.N = tf.constant(parameters.PDE_2parmeter["N"], dtype=tf.float64)
        self.m = tf.constant(parameters.PDE_2parmeter["m"], dtype=tf.float64)
        self.miu0 = tf.constant(parameters.PDE_2parmeter["miu0"], dtype=tf.float64)
        self.S = tf.constant(parameters.PDE_2parmeter["S"], dtype=tf.float64)
        self.i0 = tf.constant(parameters.PDE_2parmeter["i0"], dtype=tf.float64)
        self.z0 = tf.constant(parameters.PDE_2parmeter["z0"], dtype=tf.float64)
        self.kp = tf.constant(parameters.PDE_2parmeter["kp"], dtype=tf.float64)
        self.kv = tf.constant(parameters.PDE_2parmeter["kv"], dtype=tf.float64)
        self.g = tf.constant(parameters.PDE_2parmeter["g"], dtype=tf.float64)
        self.DG = tf.constant(parameters.PDE_2parmeter["DG"], dtype=tf.float64)
        #self.x_f_tf = tf.constant(dataSet.Trainingdata.x_f_arr, dtype=tf.float64)
        #self.par_f_tf = tf.constant(dataSet.Trainingdata.par_f_arr, dtype=tf.float64)
        #self.x_f_data = [self.par_f_tf, self.x_f_tf]
        #self.x_f_data = np.array(dataSet['x_f_data'])
        #self.dims=self.x_f_data.shape

        self.lamda0 = 3
        a=parameters.axis_par["Num_training"]


        # 创建与FFN输出相同形状的lamda
        self.lamda = tf.Variable(
            initial_value=tf.random.uniform(
                shape=(a,2),  # 与FFN输出相同形状
                minval=2,
                maxval=self.lamda0,
                dtype=tf.float64
            ),
            trainable=True,
            name='lamda',
            dtype=tf.float64
        )




###############################################################################
    @tf.function   
    def FPK_loss(self, net_pdf, net_control, x_data):
        p_f, x_f = x_data
        x1_f, x2_f = tf.split(x_f, 2, axis=1)

        # 方法1：使用持久的GradientTape计算所有导数
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x1_f)
            tape2.watch(x2_f)

            # 计算网络输出
            ux1x2 = net_pdf([p_f, tf.concat([x1_f, x2_f], 1)], training=True)
            cx1x2 = net_control([p_f, tf.concat([x1_f, x2_f], 1)], training=True)

            # 使用tape2计算一阶导数
            dudx1 = tape2.gradient(ux1x2, x1_f)
            dudx2 = tape2.gradient(ux1x2, x2_f)
            dcdx2 = tape2.gradient(cx1x2, x2_f)

        # 现在计算二阶导数
        # 注意：dudx2是在tape2中计算出来的，所以我们可以用tape2来计算它的导数
        dudx2_2 = tape2.gradient(dudx2, x2_f)

        # 清理tape2
        del tape2

        # 如果导数计算结果为None，使用0代替
        if dudx1 is None:
            dudx1 = tf.zeros_like(x1_f)
        if dudx2 is None:
            dudx2 = tf.zeros_like(x2_f)
        if dcdx2 is None:
            dcdx2 = tf.zeros_like(x2_f)
        if dudx2_2 is None:
            dudx2_2 = tf.zeros_like(x2_f)

        # 计算损失项
        part_1 = (self.kv * self.z0 ** 2 + dcdx2) /((self.z0 * x1_f + 1) ** 2)
        part_2 = (self.kp * self.z0 ** 2 * x1_f + self.kv * self.z0 ** 2 * x2_f + cx1x2 + self.i0) * ux1x2

        part_3 = ((self.kp * self.z0 ** 2 * x1_f + self.kv * self.z0 ** 2 * x2_f + cx1x2 + self.i0) / (
                    self.z0 * x1_f + 1))**2

        loss1 = 2 * self.miu0 * self.N ** 2 * self.S / (4 * self.m * self.z0 ** 2) * part_1 * part_2
        loss2 = -self.z0 ** 2 * x2_f * dudx1
        loss3 = -(self.g - self.miu0 * self.N ** 2 * self.S / (4 * self.m * self.z0 ** 2) * part_3 ) * dudx2

        # 确保dudx2_2是张量
        if not isinstance(dudx2_2, tf.Tensor):
            dudx2_2 = tf.convert_to_tensor(dudx2_2, dtype=tf.float64)

        loss4 = self.DG / (2 * self.m ** 2 * self.z0 ** 2) * dudx2_2

        loss = loss1 + loss2 + loss3 + loss4
        loss=self.lamda*loss

        return loss
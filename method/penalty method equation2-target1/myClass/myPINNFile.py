# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:24:24 2020

@author: WantaoJia
"""

import tensorflow as tf
import numpy as np
from tensorflow_probability import optimizer

loss_tracker = tf.keras.metrics.Mean(name="loss")
loss_tracker_u = tf.keras.metrics.Mean(name="lossu")
loss_tracker_f_all = tf.keras.metrics.Mean(name="lossf_all")
loss_tracker_f = tf.keras.metrics.Mean(name="lossf")
loss_tracker_f_add1 = tf.keras.metrics.Mean(name="lossf_add1")
loss_tracker_f_add2 = tf.keras.metrics.Mean(name="lossf_add2")
loss_tracker_n = tf.keras.metrics.Mean(name="lossn")
loss_tracker_c = tf.keras.metrics.Mean(name="lossc")

mae_metric_u = tf.keras.metrics.MeanAbsoluteError(name = "maeu")
mae_metric_f = tf.keras.metrics.MeanAbsoluteError(name = "maef")

tf.keras.backend.set_floatx('float64')

class myPINN_target_control(tf.keras.Model):
    def __init__(self,dataSet,parameters,myPDE,myPDF,net_pdf,net_control):
        super().__init__()
###############################################################################
        self.parameters = parameters
        self.dataSet = dataSet
###############################################################################
        self.C_par = tf.constant(self.parameters.system_par["C_par"],dtype=tf.float64)
        self.R_par = tf.constant(self.parameters.system_par["R_par"],dtype=tf.float64)
###############################################################################
        self.par_b = tf.constant(dataSet.IBCdata.par_BIC_arr,dtype=tf.float64)
        self.x_b = tf.constant(dataSet.IBCdata.x_BIC_arr,dtype=tf.float64)
        self.x_data_b = [self.par_b,self.x_b]
        self.y_b = tf.constant(dataSet.IBCdata.u_BIC_arr,dtype=tf.float64)

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

###############################################################################
    @tf.function
    def train_step(self,data):
        #data transfered
        x, y =data
        # 第一步：只更新lamda参数
        with tf.GradientTape(watch_accessed_variables=False) as tape1:
            tape1.watch([v for v in self.trainable_variables if v.name == 'lamda'])
            loss,mylossu, mylossf,mylossn,mylossc = self.loss_all(x)

        # 只获取lamda参数的梯度
        lamda_vars = [v for v in self.trainable_variables if v.name == 'lamda']
        gradients_lamda = tape1.gradient(-loss, lamda_vars)

        # 应用梯度到lamda参数
        self.optimizer.apply_gradients(zip(gradients_lamda, lamda_vars))

        # 第二步：更新其他所有参数（除了lamda）
        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch([v for v in self.trainable_variables if v.name != 'lamda'])
            loss,mylossu, mylossf,mylossn,mylossc = self.loss_all(x)

        # 获取其他参数的梯度
        other_vars = [v for v in self.trainable_variables if v.name != 'lamda']
        gradients_other = tape2.gradient(loss, other_vars)

        # 应用梯度到其他参数
        self.optimizer.apply_gradients(zip(gradients_other, other_vars))




        loss_tracker.update_state(loss)
        loss_tracker_u.update_state(mylossu)
        loss_tracker_f.update_state(mylossf)
        loss_tracker_n.update_state(mylossn)
        loss_tracker_c.update_state(mylossc)

        return {"loss": loss_tracker.result(),"lossu": loss_tracker_u.result(),
                "lossf": loss_tracker_f.result(),
                "lossn": loss_tracker_n.result(),"lossc": loss_tracker_c.result()}

    @property
    def metrics(self):
        # We list our "Metric" objects here so that 'reset_stats()' can be called
        # automatically at the start of each epoch

        return [loss_tracker, loss_tracker_u, loss_tracker_f, loss_tracker_n, loss_tracker_c]

    #   @tf.function
    def loss_all(self, x_data):
        mylossu = self.loss_data(self.x_data_b, self.y_b)
        mylossf = self.loss_residual(x_data)
        mylossn = self.loss_normalization_GL()
        mylossc = self.loss_cost_GL()
        loss = self.weight_uf * (self.weight_u * mylossu + self.weight_f * mylossf
                                     + self.weight_n * mylossn + self.weight_c * mylossc)
        return loss, mylossu, mylossf, mylossn, mylossc

        #    @tf.function
    def loss_data(self, x_data_b, y_b):
        y_pred = self.net_pdf(x_data_b, training=True)
        mylossu = tf.math.reduce_mean(tf.math.square(y_pred - y_b))

        return mylossu

        #    @tf.function
    def loss_residual(self, x_data):
        ffun = self.myPDE.FPK_loss(self.net_pdf, self.net_control, x_data)
        mylossf = tf.math.reduce_mean(tf.math.square(ffun))

        return mylossf

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

        part1_arr = (tf.math.log(pdf_pred_N_arr) - tf.math.log(self.dataSet.Costdata.targetPDF)) ** 2
        part1_mat = tf.reshape(part1_arr, self.dataSet.Costdata.Par_t1_N_mat.shape)
        int1 = tf.matmul(self.dataSet.Costdata.weight_auxi_gl_mat2arr, part1_mat)

        part2_arr = tf.reduce_sum(control_pred_N_arr, axis=1, keepdims=True) ** 2 * pdf_pred_N_arr
        part2_mat = tf.reshape(part2_arr, self.dataSet.Costdata.Par_t1_N_mat.shape)
        int2 = tf.matmul(self.dataSet.Costdata.weight_auxi_gl_mat2arr, part2_mat)

        mycost = tf.math.reduce_mean(self.C_par * int1 + self.R_par * int2)

        return mycost
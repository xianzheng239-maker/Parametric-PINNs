# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:17:18 2025

@author: lenovo
"""

import tensorflow as tf
import numpy as np

class CostPoints2D():
    def __init__(self,Auxidata,targetPDF):


        self.par_t1_lb = Auxidata.parameters.parametric_par["Q1_lb"]
        self.par_t1_ub = Auxidata.parameters.parametric_par["Q1_ub"]
        
        self.Num_par_cost = Auxidata.parameters.axis_auxi_par["Num_par_cost"]        
        
        x1_lb_auxi = Auxidata.parameters.axis_auxi_par["x1_lb_auxi"]
        x1_ub_auxi = Auxidata.parameters.axis_auxi_par["x1_ub_auxi"]
        x1_auxi_IntNum = Auxidata.parameters.axis_auxi_par["x1_auxi_IntNum"]
        x1_auxi_order = Auxidata.parameters.axis_auxi_par["x1_auxi_order"]

        x2_lb_auxi = Auxidata.parameters.axis_auxi_par["x2_lb_auxi"]
        x2_ub_auxi = Auxidata.parameters.axis_auxi_par["x2_ub_auxi"]
        x2_auxi_IntNum = Auxidata.parameters.axis_auxi_par["x2_auxi_IntNum"]
        x2_auxi_order = Auxidata.parameters.axis_auxi_par["x2_auxi_order"]

        self.x1_auxi_gl_np, self.x1_weight_auxi_gl_np = Auxidata.getMultiGLPoints(x1_lb_auxi,x1_ub_auxi,x1_auxi_IntNum,x2_auxi_order)
        self.x2_auxi_gl_np, self.x2_weight_auxi_gl_np = Auxidata.getMultiGLPoints(x2_lb_auxi,x2_ub_auxi,x2_auxi_IntNum,x2_auxi_order)

        self.myPDF = targetPDF

        self.generate_cost_GL()
        
    def generate_cost_GL(self):
        x1_auxi_gl_tf = tf.convert_to_tensor(self.x1_auxi_gl_np, dtype=tf.float64)
        x1_weight_auxi_gl_tf = tf.convert_to_tensor(self.x1_weight_auxi_gl_np, dtype=tf.float64)

        x2_auxi_gl_tf = tf.convert_to_tensor(self.x2_auxi_gl_np, dtype=tf.float64)
        x2_weight_auxi_gl_tf = tf.convert_to_tensor(self.x2_weight_auxi_gl_np, dtype=tf.float64)

        X1_auxi_gl_mat,X2_auxi_gl_mat = tf.meshgrid(x1_auxi_gl_tf,x2_auxi_gl_tf)
        x1_weight_auxi_gl_mat,x2_weight_auxi_gl_mat = tf.meshgrid(x1_weight_auxi_gl_tf,x2_weight_auxi_gl_tf)
        
        X1_auxi_gl_mat2arr = tf.reshape(X1_auxi_gl_mat,(-1,1))
        X2_auxi_gl_mat2arr = tf.reshape(X2_auxi_gl_mat,(-1,1))
        
        x1_weight_auxi_gl_mat2arr = tf.reshape(x1_weight_auxi_gl_mat,(1,-1))
        x2_weight_auxi_gl_mat2arr = tf.reshape(x2_weight_auxi_gl_mat,(1,-1))
        self.weight_auxi_gl_mat2arr = x1_weight_auxi_gl_mat2arr * x2_weight_auxi_gl_mat2arr


        #固定参数为常数，避免 par_t1 影响整体结果（目标2不依赖参数）
        #par_const = tf.constant(self.par_t1_lb, dtype=tf.float64)
        #par_t1_arr = tf.fill([self.Num_par_cost, 1], par_const)

        par_t1_arr = tf.random.uniform([self.Num_par_cost,1],
                                      minval=self.par_t1_lb,maxval=self.par_t1_ub,dtype=tf.float64)


        C_normal_list = []
        for par_t1 in par_t1_arr:
            C_temp = self.myPDF.normalize_constant_fun(self.myPDF.exp_phifun_targetfun,par_t1) 
            C_normal_list.append(C_temp)
        C_normal_arr = np.array(C_normal_list, dtype=np.float64).reshape(-1,1)
        

        self.Par_t1_N_mat,X1t_auxi_gl_mat = tf.meshgrid(par_t1_arr,X1_auxi_gl_mat2arr)
        self.C_normal_mat,_ = tf.meshgrid(C_normal_arr,X1_auxi_gl_mat2arr)
        _ ,X2t_auxi_gl_mat = tf.meshgrid(par_t1_arr,X2_auxi_gl_mat2arr)
        

        self.Par_t1_N_mat2arr = tf.reshape(self.Par_t1_N_mat,(-1,1))
        self.C_normal_mat2arr = tf.reshape(self.C_normal_mat,(-1,1))
        self.X1t_auxi_gl_mat2arr = tf.reshape(X1t_auxi_gl_mat,(-1,1))
        self.X2t_auxi_gl_mat2arr = tf.reshape(X2t_auxi_gl_mat,(-1,1))
        
        self.Par_N_mat2arr =self.Par_t1_N_mat2arr
        self.Xt_auxi_gl_mat2arr = tf.concat([self.X1t_auxi_gl_mat2arr,self.X2t_auxi_gl_mat2arr],1)

        
        self.targetPDF = tf.convert_to_tensor(
            self.myPDF.targetPDF(self.X1t_auxi_gl_mat2arr,self.X2t_auxi_gl_mat2arr,
                                 self.Par_t1_N_mat2arr,self.C_normal_mat2arr),
            dtype=tf.float64
        )
        return

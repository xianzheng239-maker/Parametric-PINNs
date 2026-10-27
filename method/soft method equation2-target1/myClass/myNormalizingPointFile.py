# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:06:25 2025

@author: lenovo
"""

import tensorflow as tf
import numpy as np

class NormalizingPoints2D():
    def __init__(self,Auxidata):

        self.par_t1_lb = Auxidata.parameters.parametric_par["Q1_lb"]
        self.par_t1_ub = Auxidata.parameters.parametric_par["Q1_ub"]
        
        self.Num_par_normalize = Auxidata.parameters.axis_auxi_par["Num_par_normalize"]

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

        self.generate_normalization_GL()

###############################################################################        
    def generate_normalization_GL(self):
        
        x1_auxi_gl_tf = self.x1_auxi_gl_np
        x1_weight_auxi_gl_tf = self.x1_weight_auxi_gl_np

        x2_auxi_gl_tf = self.x2_auxi_gl_np
        x2_weight_auxi_gl_tf = self.x2_weight_auxi_gl_np

        X1_auxi_gl_mat,X2_auxi_gl_mat = tf.meshgrid(x1_auxi_gl_tf,x2_auxi_gl_tf)
        x1_weight_auxi_gl_mat,x2_weight_auxi_gl_mat = tf.meshgrid(x1_weight_auxi_gl_tf,x2_weight_auxi_gl_tf)
        
        X1_auxi_gl_mat2arr = tf.reshape(X1_auxi_gl_mat,(-1,1))
        X2_auxi_gl_mat2arr = tf.reshape(X2_auxi_gl_mat,(-1,1))
        
        x1_weight_auxi_gl_mat2arr = tf.reshape(x1_weight_auxi_gl_mat,(1,-1))
        x2_weight_auxi_gl_mat2arr = tf.reshape(x2_weight_auxi_gl_mat,(1,-1))
        self.weight_auxi_gl_mat2arr = x1_weight_auxi_gl_mat2arr * x2_weight_auxi_gl_mat2arr
        

        par_t1_arr = tf.random.uniform([self.Num_par_normalize,1],
                                      minval=self.par_t1_lb,maxval=self.par_t1_ub,dtype=tf.float64)

        self.Par_t1_N_mat,X1t_auxi_gl_mat= tf.meshgrid(par_t1_arr,X1_auxi_gl_mat2arr)
        _ ,X2t_auxi_gl_mat = tf.meshgrid(par_t1_arr,X2_auxi_gl_mat2arr)
        

        self.Par_t1_N_mat2arr = tf.reshape(self.Par_t1_N_mat,(-1,1))
        self.X1t_auxi_gl_mat2arr = tf.reshape(X1t_auxi_gl_mat,(-1,1))
        self.X2t_auxi_gl_mat2arr = tf.reshape(X2t_auxi_gl_mat,(-1,1))
        
        self.Par_N_mat2arr = self.Par_t1_N_mat2arr
        self.Xt_auxi_gl_mat2arr = tf.concat([self.X1t_auxi_gl_mat2arr,self.X2t_auxi_gl_mat2arr],1)
        
        return
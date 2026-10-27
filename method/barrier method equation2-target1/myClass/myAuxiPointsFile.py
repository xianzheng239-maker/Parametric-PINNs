# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 22:07:59 2020

@author: WantaoJia
"""

import numpy as np


class AuxiMultiGL2D():

    def __init__(self, parameters):
        self.parameters = parameters

###############################################################################
    def getMultiGLPoints(self,x_lb_auxi,x_ub_auxi,x_auxi_IntNum,x_auxi_order):
        
        Interval_arr = np.linspace(x_lb_auxi,x_ub_auxi,x_auxi_IntNum+1).reshape([-1,1]).astype(np.float64)
        
        x_auxi_sub_gl_list=[]
        weight_auxi_sub_gl_list=[]
        
        for index_traning_point in range(x_auxi_IntNum):
            
            x_auxi_sub_lb = Interval_arr[index_traning_point]
            x_auxi_sub_ub = Interval_arr[index_traning_point+1]
            
            x_auxi_sub_gl_arr, weight_auxi_sub_gl_arr = self.GenerateGLPoint(x_auxi_order,x_auxi_sub_lb,x_auxi_sub_ub)
            
            x_auxi_sub_gl_list.append(x_auxi_sub_gl_arr)
            weight_auxi_sub_gl_list.append(weight_auxi_sub_gl_arr)
        x_auxi_gl_np = np.vstack(x_auxi_sub_gl_list).astype(np.float64)
        weight_auxi_gl_np = np.vstack(weight_auxi_sub_gl_list).astype(np.float64)

            
        return x_auxi_gl_np, weight_auxi_gl_np

###############################################################################        
    def GenerateGLPoint(self,Num_gl,a_gl,b_gl):
        
        gl_x, gl_w = np.polynomial.legendre.leggauss(Num_gl)
        gl_x = gl_x.reshape([-1,1]).astype(np.float64)
        gl_w = gl_w.reshape([-1,1]).astype(np.float64)
        
        gl = ((b_gl-a_gl)*gl_x+a_gl+b_gl)/2.0
        weight_gl = 0.5*(b_gl-a_gl)*gl_w
        
        return gl.astype(np.float64),weight_gl.astype(np.float64)

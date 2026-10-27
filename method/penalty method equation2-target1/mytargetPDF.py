# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:38:26 2023

@author: WantaoJia
"""

import numpy as np
import tensorflow as tf
import scipy as sc
from scipy import integrate

class mytarget():
    def __init__(self,parameters):
###############################################################################
        self.a_par = tf.constant(parameters.target_par["a_par"],dtype=tf.float64)
        self.b_par = tf.constant(parameters.target_par["b_par"],dtype=tf.float64)
        #self.omega1 = tf.constant(parameters.target_par["omega1"],dtype=tf.float64)
###############################################################################                
        self.pi = tf.constant(np.pi,dtype=tf.float64)
###############################################################################
        self.x1_lb = parameters.axis_par["x1_lb"]
        self.x1_ub = parameters.axis_par["x1_ub"]
        self.x2_lb = parameters.axis_par["x2_lb"]
        self.x2_ub = parameters.axis_par["x2_ub"]
##################################################################################################################
        #self.C_normal_target = self.normalize_constant_fun(self.exp_phifun_targetfun)
        self.test = None
################################################################################
    def normalize_constant_fun(self,exp_phi,par_t1):
        C_temp = sc.integrate.dblquad(exp_phi,self.x2_lb, self.x2_ub, self.x1_lb, self.x1_ub, args=(par_t1,))
        #scipy.integrate.dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-08, epsrel=1.49e-08)
        #从 x = a..b 和 y = gfun(x)..hfun(x) 返回 func(y, x) 的双(定)积分。
        #func： 可调用的，至少有两个变量的 Python 函数或方法：y 必须是第一个参数，x 必须是第二个参数。
        return C_temp[0]
################################################################################  目标1
    #def phifun_targetfun(self,x,y,par_t1):
    #    phifun_target = self.a_par*(y**2+par_t1**2*x**2)- self.b_par*(y**2+par_t1**2*x**2)**2
    #    return phifun_target

################################################################################  目标1
    def phifun_targetfun(self, x, y, par_t1):
        phifun_target = -0.5 * par_t1 * (500 * x ** 2 + y ** 2)
        return phifun_target
    
    def exp_phifun_targetfun(self,x,y,par_t1):
        exp_phifun_target = np.exp(self.phifun_targetfun(x,y,par_t1))
        return exp_phifun_target
    
    def targetPDF(self,x,y,par_t1,C_normal_target):
        pdf_garget=self.exp_phifun_targetfun(x,y,par_t1)/C_normal_target
        return pdf_garget
    
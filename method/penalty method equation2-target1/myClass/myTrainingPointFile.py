# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 21:28:44 2020

@author: WantaoJia
"""

import numpy as np

class TrainingPoints2D():
    def __init__(self,parameters):
        
        self.x1_lb = parameters.axis_par["x1_lb"]
        self.x1_ub = parameters.axis_par["x1_ub"]
        
        self.x2_lb = parameters.axis_par["x2_lb"]
        self.x2_ub = parameters.axis_par["x2_ub"]
        


        self.par_t1_lb = parameters.parametric_par["Q1_lb"]
        self.par_t1_ub = parameters.parametric_par["Q1_ub"]
        
        self.Num_training = parameters.axis_par["Num_training"]

        self.GenerateTrainingPoints2DRandom()
        
    def GenerateTrainingPoints2DRandom(self):

        self.x1_f_arr = np.random.uniform(self.x1_lb,self.x1_ub,[self.Num_training,1])
        self.x2_f_arr = np.random.uniform(self.x2_lb ,self.x2_ub,[self.Num_training,1])
        self.x_f_arr = np.hstack((self.x1_f_arr,self.x2_f_arr))
        

        self.par_t1_f_arr = np.random.uniform(self.par_t1_lb,self.par_t1_ub,[self.Num_training,1])
        self.par_f_arr = self.par_t1_f_arr
   
        self.u_f_arr = np.zeros(self.x1_f_arr.shape)
     
        return
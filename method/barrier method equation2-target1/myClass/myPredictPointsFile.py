# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:24:24 2020

@author: WantaoJia
"""

import numpy as np

class predictionPoin2D():
    def __init__(self,parameters):
        
        self.x1_pred_lb = parameters.axis_par["x1_pred_lb"]
        self.x1_pred_ub = parameters.axis_par["x1_pred_ub"]
        self.x2_pred_lb = parameters.axis_par["x2_pred_lb"]
        self.x2_pred_ub = parameters.axis_par["x2_pred_ub"]
        self.x1_pred_step = parameters.axis_par["x1_pred_step"]
        self.x2_pred_step = parameters.axis_par["x2_pred_step"]

        self.par_t1_lb = parameters.parametric_par["Q1_lb"]
        self.par_t1_ub = parameters.parametric_par["Q1_ub"]
        
        self.num_par_pred = parameters.parametric_par["num_par_pred"]
############################################################################################                
       
        self.GeneratePredPoints()
        
    def GeneratePredPoints(self):

        
        par_t1_arr = np.linspace(self.par_t1_lb,self.par_t1_ub,self.num_par_pred,
                                 endpoint=True, 
                                 #retstep=False, dtype=None, axis=0
                                 ).reshape(-1,1).astype(np.float64)
        print(par_t1_arr)
        
        x1_pred_arr = np.arange(self.x1_pred_lb,self.x1_pred_ub+self.x1_pred_step,self.x1_pred_step, dtype=np.float64)
        x2_pred_arr = np.arange(self.x2_pred_lb,self.x2_pred_ub+self.x2_pred_step,self.x2_pred_step, dtype=np.float64)
        


        self.Par_t1_pred_3d_mat,self.X1_pred_3d_mat,self.X2_pred_3d_mat= np.meshgrid(par_t1_arr,x1_pred_arr, x2_pred_arr)
        self.Par_t1_pred_3d_mat = self.Par_t1_pred_3d_mat.astype(np.float64)
        self.X1_pred_3d_mat = self.X1_pred_3d_mat.astype(np.float64)
        self.X2_pred_3d_mat = self.X2_pred_3d_mat.astype(np.float64)
        

        self.Par_t1_pred_3d_mat2arr = self.Par_t1_pred_3d_mat.reshape([-1,1])
        self.Par_pred_3d_mat2arr = self.Par_t1_pred_3d_mat2arr
        
        self.X1_pred_3d_mat2arr = self.X1_pred_3d_mat.reshape([-1,1])
        self.X2_pred_3d_mat2arr = self.X2_pred_3d_mat.reshape([-1,1])
        self.X_pred_3d_mat2arr = np.hstack((self.X1_pred_3d_mat2arr,self.X2_pred_3d_mat2arr)).astype(np.float64)
       
        return
            

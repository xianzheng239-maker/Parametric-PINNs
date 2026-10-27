# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:24:24 2020

@author: WantaoJia
"""

import numpy as np

class boudaryAndInitial2D():
    def __init__(self,parameters):
             
        self.x1_lb = parameters.axis_par["x1_lb"]
        self.x1_ub = parameters.axis_par["x1_ub"]
               
        self.x2_lb = parameters.axis_par["x2_lb"]
        self.x2_ub = parameters.axis_par["x2_ub"]
        


        self.par_t1_lb = parameters.parametric_par["Q1_lb"]
        self.par_t1_ub = parameters.parametric_par["Q1_ub"]
        
        self.Num_bound = parameters.axis_par["Num_bound"]
############################################################################################                
        self.x1_BIC_arr = None
        self.x2_BIC_arr = None
        self.u_BIC_arr = None
############################################################################################        
        self.GenerateBICPoints()

    def GenerateBICPoints(self):
        #####################################################################################
        self.x1_BIC_arr,self.x2_BIC_arr, self.u_BIC_arr = self.GenerateBoundaryPointsRandom()
        self.x_BIC_arr = np.hstack((self.x1_BIC_arr,self.x2_BIC_arr))
        

        self.par_t1_BIC_arr = np.random.uniform(self.par_t1_lb,self.par_t1_ub,[self.Num_bound,1])
        self.par_BIC_arr = self.par_t1_BIC_arr
        
        return
    
    def GenerateBoundaryPointsRandom(self):
        length_x1 = self.x1_ub - self.x1_lb
        length_x2 = self.x2_ub - self.x2_lb
        
        x1_boundary_arr = np.zeros([self.Num_bound,1])
        x2_boundary_arr = np.zeros([self.Num_bound,1])
        
        position_arr = np.random.uniform(0.0,2.0*(length_x1+length_x2),[self.Num_bound,1])
        
        for i_index in range(self.Num_bound):
            position = position_arr[i_index]
            
            if position <= length_x1:
                x1_boundary_arr[i_index] = position + self.x1_lb
                x2_boundary_arr[i_index] = self.x2_ub
            elif position <= length_x2 + length_x1:
                x1_boundary_arr[i_index] = self.x1_ub
                x2_boundary_arr[i_index] = self.x2_ub - (position - length_x1)
            elif position <= 2.0*length_x1+length_x2:
                x1_boundary_arr[i_index] = self.x1_ub - (position - length_x1 - length_x2)
                x2_boundary_arr[i_index] = self.x2_lb
            else:
                x1_boundary_arr[i_index] = self.x1_lb
                x2_boundary_arr[i_index] = (position - 2.0*length_x1 - length_x2) + self.x2_lb
        
        u_boundary_arr = np.zeros(x1_boundary_arr.shape)
        
        return x1_boundary_arr,x2_boundary_arr,u_boundary_arr
                


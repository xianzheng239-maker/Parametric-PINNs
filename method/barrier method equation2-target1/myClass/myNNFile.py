# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 20:41:54 2020

@author: WantaoJia
"""

import tensorflow as tf
import numpy as np
from .myLayer import Finetuningnet,MyDenseLayer

tf.keras.backend.set_floatx('float64')

class myNN_paramteric(tf.keras.Model):
    def __init__(self,NN_par,parameters):
        super().__init__()
        self.net_p = myNNAuto_subnet(NN_par["net_p"])
        self.net_c = myNNAuto_subnet(NN_par["net_c"])
        self.net_g = myNNAuto_subnet(NN_par["net_g"])
        self.concat = tf.keras.layers.Concatenate(axis=-1) 
        
        self.p_l = np.array(parameters.parametric_par["Q1_lb"], dtype=np.float64)
        self.p_u = np.array(parameters.parametric_par["Q1_ub"], dtype=np.float64)
        
        self.x_l = np.array([parameters.axis_par["x1_lb"],parameters.axis_par["x2_lb"]], dtype=np.float64)
        self.x_u = np.array([parameters.axis_par["x1_ub"],parameters.axis_par["x2_ub"]], dtype=np.float64)

    @tf.function
    def call(self, input):
        p,x= input
        
        p = 2.0*(p-self.p_l)/(self.p_u-self.p_l)-1.0
        hp = self.net_p(p)
        
        x = 2.0*(x-self.x_l)/(self.x_u-self.x_l)-1.0
        hc = self.net_c(x)
        
        hconcat = self.concat([hp,hc])
        
        output = self.net_g(hconcat)
        return output
    
class myNNAuto_subnet(tf.keras.Model):
    def __init__(self,NN_parameters):
        super().__init__()
        self.NN_parameters = NN_parameters
        self.NN_net_list = self.NN_layers(NN_parameters)
        temp = 0.0
    
    def call(self, input):
        
        x = input
        
        for i_layer in range(0,self.NN_parameters["NumOflayer"]):
            x = self.NN_net_list[i_layer](x)
            
        output = x
        return output

    def NN_layers(self,NN_parameters):
        
        NN_net_list = []
        
        for i_layer in range(0,NN_parameters["NumOflayer"]):
            
            numOfNeronsOfthislayer =  NN_parameters["NumOfNeuronEachLayer"][i_layer]
            activefun = NN_parameters["ActivateFunOfEachLayer"][i_layer]
            
            denselayer = tf.keras.layers.Dense(units=numOfNeronsOfthislayer,
                                               activation=activefun)
            NN_net_list.append(denselayer)
            
        return NN_net_list
    

##############################################################################################################################################
class myNN_finetuning(tf.keras.Model):
    def __init__(self,parameters,loadednet):
        super().__init__()
        

        self.net_g_finetuning = Finetuningnet(loadednet.net_g)
        
        self.net_p_finetuning = loadednet.net_p
 
        self.net_c_finetuning = loadednet.net_c

        self.concat = tf.keras.layers.Concatenate(axis=-1) 
        
        self.p_l = np.array(parameters.parametric_par["omega_lb"], dtype=np.float64)
        self.p_u = np.array(parameters.parametric_par["omega_ub"], dtype=np.float64)
        
        self.x_l = np.array([parameters.axis_par["x1_lb"],parameters.axis_par["x2_lb"]], dtype=np.float64)
        self.x_u = np.array([parameters.axis_par["x1_ub"],parameters.axis_par["x2_ub"]], dtype=np.float64)
       
        temp = 0.0
    @tf.function
    def call(self, inputs):
        
        p,x= inputs
        
        p = 2.0*(p-self.p_l)/(self.p_u-self.p_l)-1.0
        hp = self.net_p_finetuning(p)
        
        x = 2.0*(x-self.x_l)/(self.x_u-self.x_l)-1.0
        hc = self.net_c_finetuning(x)
        
        hconcat = self.concat([hp,hc])
        
        output = self.net_g_finetuning(hconcat)
        return output    
 
        

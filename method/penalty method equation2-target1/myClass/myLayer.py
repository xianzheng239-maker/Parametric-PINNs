# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:10:44 2024

@author: jiawa
"""

import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform, GlorotNormal

tf.keras.backend.set_floatx('float64')

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self,weights,bias,activation):
        super(MyDenseLayer, self).__init__()
        s, u, v = tf.linalg.svd(weights)
    
        self.u = u
        self.v = v
        self.b = tf.Variable(tf.zeros(bias.shape,dtype=tf.float64),dtype=tf.float64, trainable=True,name = 'b')
        self.activation = activation
    
        self.s = tf.Variable(tf.random.uniform(shape= s.shape,minval=0.0, maxval=1.0,dtype=tf.float64),
                         dtype=tf.float64, trainable=True,name = 's') #True or False
    
        temp = 0.0
        
    @tf.function
    def call(self, inputs):
        dense = tf.matmul(inputs, tf.matmul(self.u, tf.matmul(tf.linalg.diag(tf.math.softplus(self.s)), self.v, adjoint_b=True))) + self.b
        output = self.activation(dense)
        return output

class Finetuningnet(tf.keras.layers.Layer):
    def __init__(self,loadedmodel):
        super(Finetuningnet, self).__init__()
    
        self.loadedmodel = loadedmodel
        
        self.numberlayer = len(loadedmodel.layers)
        self.NN_net_list = self.NN_layers()
        
        temp = 0.0
        
    @tf.function
    def call(self, input):
        
        x = input
        
        for i_layer in range(0,self.numberlayer ):
            x = self.NN_net_list[i_layer](x)
            
        output = x
        
        return output
  
    def NN_layers(self):
        
        NN_net_list = []
        
        layer = self.loadedmodel.layers[0]
        layer.trainable=False
        NN_net_list.append(layer)
             
        for i_layer in range(1,self.numberlayer-1):
            
            layer = MyDenseLayer(self.loadedmodel.layers[i_layer].weights[0],
                                 self.loadedmodel.layers[i_layer].weights[1],
                                 self.loadedmodel.layers[i_layer].activation)
            NN_net_list.append(layer)
            
        layer = self.loadedmodel.layers[-1]
        layer.trainable=False
        NN_net_list.append(layer)
        return NN_net_list
        
    
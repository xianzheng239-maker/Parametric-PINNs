# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 12:15:32 2020

@author: WantaoJia
"""

import tensorflow as tf

from datetime import datetime
from myClass.myFunctionforLBFGs import Function_LBFGS

class myML():
    def __init__(self,dataSet,parameters,myPDE,myPINNmodel):
        #######################################################################
        #machine learning parameter
        self.batch_size = parameters.ml_parameter["batch_size"]
        self.epochs = parameters.ml_parameter["epochs"]
        self.verbose = parameters.ml_parameter["verbose"]
        self.patience = parameters.ml_parameter["patience"]
        self.lr_schedule = parameters.ml_parameter["learning_rate_schedule"]
        #######################################################################
        self.x_f_tf = tf.constant(dataSet.Trainingdata.x_f_arr,dtype=tf.float64)
        self.par_f_tf = tf.constant(dataSet.Trainingdata.par_f_arr,dtype=tf.float64)
        self.u_t_tf = tf.constant(dataSet.Trainingdata.u_f_arr,dtype=tf.float64)
                
        self.x_f_data = [self.par_f_tf,self.x_f_tf]
        self.y_f_data = self.u_t_tf
        #######################################################################
        self.myPINNmodel = myPINNmodel
        self.history = None
        self.F_lbfgs = Function_LBFGS()
        
        self.verbose_callback = CustomVerboseCallback(parameters)
        
    def modelTrain(self):
        self.myPINNmodel.compile(optimizer=tf.keras.optimizers.Adam())
            
        self.history=self.myPINNmodel.fit(
            self.x_f_data,self.y_f_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=[
                    #tf.keras.callbacks.EarlyStopping(monitor='loss',patience=self.patience),
                    tf.keras.callbacks.LearningRateScheduler(self.lr_schedule),
                    self.verbose_callback
                    ]
                )
        
    def modelTrain_lbfgs(self):
        self.history_lbfgs = self.F_lbfgs.train_lbfgs(self.myPINNmodel, 
                          self.myPINNmodel.loss_all, 
                          self.x_f_data, 
                          5000)

        
    def modelPrediction(self,p,x_data):
            
            x_data = [p,x_data]
            
            y_pred=self.myPINNmodel.net_pdf(x_data)
            return y_pred

class CustomVerboseCallback(tf.keras.callbacks.Callback):
    def __init__(self,parameters):
        super().__init__()
        self.verbose_frequence = parameters.ml_parameter["verbose_frequence"]
        
    def on_train_begin(self, logs=None):
        self.start_time = datetime.now()
        self.step = 0
 
    def on_epoch_begin(self, epoch, logs=None):
        #self.epoch_start_time = datetime.now()
        pass
 
    def on_epoch_end(self, epoch, logs=None):
        self.step += 1
        if self.step % self.verbose_frequence == 0:
            print(f"Step: {self.step}, loss: {logs['loss']}")
            print(f"lossu: {logs['lossu']},lossn: {logs['lossn']},lossc: {logs['lossc']}")
            print(f"lossf: {logs['lossf']}")
            now = datetime.now()
            duration = now - self.start_time
            self.start_time = now
            print(f"Time since epoch start: {duration.total_seconds()} seconds")

            print("======================================================================")

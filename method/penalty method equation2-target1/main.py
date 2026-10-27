# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:24:24 2020

@author: W antaoJia
"""

import tensorflow as tf
import scipy.io as scio

from myClass.myNNFile import myNN_paramteric,myNN_finetuning
from myClass.myMLFile import myML
from myClass.myPINNFile import myPINN_target_control

from myClass.myGenerateDataSet import myDataSet2D
from myPDEs import myPoissonFPKSDOF
from mytargetPDF import mytarget

import parameters

from utilities import train_ML_save_Datas

import time
start_time = time.time()

################################################################################
targetPDF = mytarget(parameters)
################################################################################
dataSet = myDataSet2D(parameters,targetPDF)
################################################################################
FPKequation=myPoissonFPKSDOF(parameters,dataSet)
################################################################################
IBCdatasave='initial.mat'
scio.savemat(IBCdatasave,{'X1_arr': dataSet.IBCdata.x1_BIC_arr,
                          'X2_arr': dataSet.IBCdata.x2_BIC_arr,
                          'u_arr': dataSet.IBCdata.u_BIC_arr})
################################################################################
savepath = {
    "model_pdf":{"net":"saved/NN/myPDF"},
    "model_control":{"net":"saved/NN/myControl"}
            }

data_name = "_original"

net_pdf = myNN_paramteric(parameters.net_pdf_parametric,parameters)
net_control = myNN_paramteric(parameters.net_control_parametric,parameters)
myPINNmodel = myPINN_target_control(dataSet,parameters,FPKequation,targetPDF,
                                    net_pdf,net_control)

myMLModel=myML(
    dataSet, parameters, FPKequation,myPINNmodel
    )



train_ML_save_Datas(myMLModel,dataSet,savepath,data_name)
end_time = time.time()
total_time = end_time - start_time
print(f"总训练时间: {total_time:.2f} 秒")

################################################################################
#savepath_finetuning = {
#    "model":{"net":"saved/NN_finetuning/myNN"}
#            }
#data_name_finetuning = "_finetuning"

# loadednet = tf.keras.models.load_model(savepath["model"]["net"])
# loadednet.trainable = False 

# net_finetuning = myNN_finetuning(parameters,loadednet)

# myMLModel_fineturning = myML(
#     dataSet, parameters, myPoissonFPKequation,net_finetuning
#     )

# train_ML_save_Datas(net_finetuning,myMLModel_fineturning,dataSet,savepath_finetuning,data_name_finetuning)
################################################################################
#savepath_loaded = {
#    "model":{"net":"saved/NN_loaded/myNN"}
#            }
#data_name_loaded = "_loaded"

#loadednet = tf.keras.models.load_model(savepath_finetuning["model"]["net"])

# loadednet = tf.keras.models.load_model(savepath_loaded["model"]["net"])


#myMLModel_loaded = myML(
#    dataSet, parameters, myPoissonFPKequation,loadednet
#    )

#train_ML_save_Datas(loadednet,myMLModel_loaded,dataSet,savepath_loaded,data_name_loaded)
################################################################################


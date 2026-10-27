# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 17:46:44 2024

@author: lenovo
"""
import numpy as np
import tensorflow as tf
import scipy.io as scio
import time

def train_ML_save_Datas(MLnet,dataSet,savepath,data_name):
    time_start=time.time()

    MLnet.modelTrain()
    history=MLnet.history
    
    MLnet.modelTrain_lbfgs()
    history_lbfgs=MLnet.history_lbfgs
    
    y_f_pred=MLnet.modelPrediction(
        dataSet.Predictdata.Par_pred_3d_mat2arr,
        dataSet.Predictdata.X_pred_3d_mat2arr,
        )

    u_f_pred_tf=tf.reshape(y_f_pred,dataSet.Predictdata.X1_pred_3d_mat.shape)

          
    control_arr_pred = MLnet.myPINNmodel.net_control([dataSet.Predictdata.Par_pred_3d_mat2arr,
            dataSet.Predictdata.X_pred_3d_mat2arr])
    control_R_arr,control_I_arr=tf.split(control_arr_pred,2,axis=1)
    control_R=tf.reshape(control_R_arr,dataSet.Predictdata.X1_pred_3d_mat.shape)
    control_I=tf.reshape(control_I_arr,dataSet.Predictdata.X1_pred_3d_mat.shape)
    print(control_arr_pred.shape)
    
    control = control_R + control_I
    
    time_end=time.time()

    time_spend=time_end-time_start

    savePDFData = "pdf_mat"+data_name+".mat"
    scio.savemat(savePDFData, {'X1_mat':dataSet.Predictdata.X1_pred_3d_mat,
                                'X2_mat':dataSet.Predictdata.X2_pred_3d_mat,
                                'par_t1':dataSet.Predictdata.Par_t1_pred_3d_mat,
                                'pdf_mat':u_f_pred_tf.numpy(),
                               'y_f_pred':y_f_pred.numpy(),
                               'control_arr_pred':control_arr_pred})

    savePDFData = "control_mat"+data_name+".mat"
    scio.savemat(savePDFData, {'X1_mat':dataSet.Predictdata.X1_pred_3d_mat,
                                'X2_mat':dataSet.Predictdata.X2_pred_3d_mat,
                                'par_t1':dataSet.Predictdata.Par_t1_pred_3d_mat,
                                'control_mat':control.numpy(),
                                'control_R_mat':control_R.numpy(),
                                'control_I_mat':control_I.numpy(),
                               'y_f_pred': y_f_pred.numpy(),
                               'control_arr_pred': control_arr_pred
                                })

    saveHistory="history"+data_name+".mat"
    scio.savemat(saveHistory,{'loss':np.array(history.history["loss"]+history_lbfgs.loss),
                                  'lossu':np.array(history.history["lossu"]+history_lbfgs.lossu),
                                  'lossf':np.array(history.history["lossf"]+history_lbfgs.lossf),
                                  'lossn':np.array(history.history["lossn"]+history_lbfgs.lossn),
                                  })

    timeSpendData="time_spend"+data_name+".mat"
    scio.savemat(timeSpendData,{'time_spend':time_spend,'time_start':time_start,'time_end':time_end})

    # net.save_weights(savepath["weights"]["net"])
    # net.net_p.save_weights(savepath["weights"]["net_p"])
    # net.net_c.save_weights(savepath["weights"]["net_c"])
    # net.net_g.save_weights(savepath["weights"]["net_g"])
    
    MLnet.myPINNmodel.net_pdf.save(savepath["model_pdf"]["net"])
    MLnet.myPINNmodel.net_control.save(savepath["model_control"]["net"])
    # net.net_p.save(savepath["model"]["net_p"])
    # net.net_c.save(savepath["model"]["net_c"])
    # net.net_g.save(savepath["model"]["net_g"])
    
    return 

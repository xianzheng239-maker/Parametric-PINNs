# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 12:18:27 2020

@author: WantaoJia
"""
import math

import tensorflow as tf

tf.keras.backend.set_floatx('float64')


PDE_2parmeter={
    "N": 450,
    "m": 700,
    "miu0": 4*math.pi*1e-7,
    "S": 0.024,
    "i0": 21.2075,
    "z0": 0.01,
    "kp": 4823.043,
    "kv": 76.4208,
    "g": 9.81,
    "DG": 1.0,

}


system_par = {
    # 坐标信息
    "alpha1": -0.08,
    "beta1": 0.1,
    "R_par": 0.0,
    "C_par": 1.0
}

target_par = {
    "a_par": 1.0,
    "b_par": 1.5,
    "omega1": 1.2
}

parametric_par={
    "Q1_lb":4.8,
    "Q1_ub":5.3,
    "num_par_pred": 6
    }

noise_par = {
    "D11": 0.0001,
    "S11": 0.0001,
}

axis_par = {
    "Num_bound": 1000,
    "Num_training": 10000,

    # 坐标信息
    "x1_lb": -0.2, "x1_ub": 0.2,
    "x2_lb": -4, "x2_ub": 4,
    "x1_f_step": 0.005,
    "x2_f_step": 0.02,

    # 预测信息
    "x1_pred_lb": -0.2, "x1_pred_ub": 0.2,
    "x2_pred_lb": -4, "x2_pred_ub": 4,
    "x1_pred_step": 0.005,
    "x2_pred_step": 0.02,
}

axis_auxi_par = {
    "Num_par_normalize": 10,
    "Num_par_cost": 20,
    "x1_lb_auxi": axis_par["x1_lb"],
    "x1_ub_auxi": axis_par["x1_ub"],
    "x1_auxi_IntNum": 3,
    "x1_auxi_order": 3,

    "x2_lb_auxi": axis_par["x2_lb"],
    "x2_ub_auxi": axis_par["x2_ub"],
    "x2_auxi_IntNum": 3,
    "x2_auxi_order": 3,
}

lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[
        5000,
        10000,
        15000,
    ],
    values=[
        0.0005,
        0.0001,
        0.00005,
        0.00005,
    ]
)

ml_parameter = {
    "batch_size": axis_par["Num_training"],
    "buffer_size": axis_par["Num_training"],
    "epochs": 10000,
    "verbose": 0,
    "verbose_frequence": 100,
    "learning_rate": 0.0005,
    "learning_rate_schedule": lr_schedule,
    "decay": 1000,
    "patience": 50,
    "weight_f": 1.0,
    "weight_l1": 5.0,
    "weight_l2": 1.0,
    "weight_u": 0.0,
    "weight_n": 1.0,
    "weight_c": 5.0,
    "weight_uf": 1.0,
    "trainable_variables": None
}

net_pdf_parametric = {
    "net_p": {
        "NumOflayer": 3,
        "NumOfNeuronEachLayer": [64, 64, 64],
        "ActivateFunOfEachLayer": [tf.keras.activations.tanh,
                                   tf.keras.activations.tanh,
                                   tf.keras.activations.tanh]
    },
    "net_c": {
        "NumOflayer": 3,
        "NumOfNeuronEachLayer": [64, 64, 64],
        "ActivateFunOfEachLayer": [tf.keras.activations.tanh,
                                   tf.keras.activations.tanh,
                                   tf.keras.activations.tanh]
    },
    "net_g": {
        "NumOflayer": 5,
        "NumOfNeuronEachLayer": [128, 128, 128, 128, 1],
        "ActivateFunOfEachLayer": [tf.keras.activations.tanh,
                                   tf.keras.activations.tanh,
                                   tf.keras.activations.tanh,
                                   tf.keras.activations.tanh,
                                   tf.keras.activations.exponential]
    }
}

net_control_parametric = {
    "net_p": {
        "NumOflayer": 3,
        "NumOfNeuronEachLayer": [64, 64, 64],
        "ActivateFunOfEachLayer": [tf.keras.activations.tanh,
                                   tf.keras.activations.tanh,
                                   tf.keras.activations.tanh]
    },
    "net_c": {
        "NumOflayer": 2,
        "NumOfNeuronEachLayer": [64, 64],
        "ActivateFunOfEachLayer": [tf.keras.activations.tanh,
                                   tf.keras.activations.tanh]
    },
    "net_g": {
        "NumOflayer": 4,
        "NumOfNeuronEachLayer": [64, 64, 64, 2],
        "ActivateFunOfEachLayer": [tf.keras.activations.tanh,
                                   tf.keras.activations.tanh,
                                   tf.keras.activations.tanh,
                                   None]
    }
}





#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 02:29:50 2018

@author: loktar
"""

from __future__ import print_function
import sys
sys.dont_write_bytecode = True

import tensorflow as tf


def _Momentum(lr, momentum=0.1, use_nesterov=True):
    return tf.train.MomentumOptimizer(lr, momentum=momentum, use_nesterov=use_nesterov)

def _Adam(lr, beta1=0.9, beta2=0.999, epsilon=1e-08):
    return tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)

def _Adadelta(lr, decay=0.9, epsilon=1e-08):
    with tf.variable_scope("adadelta"):
        return tf.train.AdadeltaOptimizer(learning_rate=lr, rho=decay, epsilon=epsilon)

def _RMSProp(lr, momentum=0, decay=0.9, epsilon=1e-08):
    with tf.variable_scope("rmsprop"):
        return tf.train.RMSPropOptimizer(lr, decay=decay, momentum=momentum, epsilon=epsilon)

def get_optimizer(optimizer_name):
    optimizer_lst = {
                        "rmsprop":   _RMSProp,
                        "momentum":  _Momentum,
                        "adam":      _Adam,
                        "adadelta":  _Adadelta,
                    }
    return optimizer_lst[optimizer_name]

def _exponential_decay(start_lr, global_step, decay_steps, decay_rate, staircase=False):
    return tf.train.exponential_decay(start_lr, global_step, decay_steps, decay_rate, staircase=staircase)

def get_decay_category(category_name):
    decay_category_lst = {
                            "exponential":  _exponential_decay
                         }
    return decay_category_lst[category_name]

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 02:29:50 2018

@author: loktar
"""

import os

if __name__ == '__main__':
    lr = 0.00001
    gpu_id = 0
    os.system("python trainer.py \
               --optimizer adam \
               --start_lr {} \
               --gpu_id {} \
               $@".format(lr, str(gpu_id)))

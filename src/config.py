#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""
import os

config_common = {'input_shape': (32, 32, 3), 'num_classes': 10}

config_preprocessing = {'mean_pixel':[0, 0, 0], 'max_pixel': 255.}

config_data_loader = {'input_handle': ['../output/tfrecord/cifar10/train.tfrecord'],
                       'mode': 'tfrecord',
                       'num_repeat': 1,
                       'shuffle': True,
                       'batch_size': 128,
                       'buffer_size': 30000,
                       'num_processors': 32,
                       'augmentation': True,
                       'name': 'train_dataloader'}

config_valid_loader = {'input_handle': ['../output/tfrecord/cifar10/valid.tfrecord'],
                       'mode': 'tfrecord',
                       'num_repeat': 1,
                       'shuffle': False,
                       'batch_size': 1024,
                       'buffer_size': 100,
                       'num_processors': 32,
                       'augmentation': False,
                       'name': 'valid_dataloader'}


config_trainer = {'num_epochs': 300,
                  'step_per_epoch': 700,
                  'save_steps': 1000,
                  'device': 'gpu:0',
                  'model_name': 'resnet-101',
                  'save_path': '../output/train_results'}


config_DataAugmentation = {'random_shift_switch': True,
                                # random shift augmentaiton

                           'random_color_switch': True,
                                # random color augmentation
                                'max_hue': 0.3,
                                'max_bri': 60. / 255.,
                                'lower_sat': 0.5, 'upper_sat': 2.5,
                                'lower_con': 0.5, 'upper_con': 1.5,

                           'random_crop_switch': True,
                                # random crop augmentation
                                'resize_shorter_edge': 36,

                           'random_noise_switch': True,
                                # random noise augmentation
                                'gaussian_noise_scale': 0.01,
                                'salt_noise_scale': 0.01,
                                'pepper_noise_scale': 0.01}



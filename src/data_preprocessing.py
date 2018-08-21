#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops, math_ops
import sys
sys.dont_write_bytecode = True

from config import config_common, config_preprocessing, config_DataAugmentation

def preprocessing_func(image, label):
    """image preprocessing

    (image_array - mean) / max_value(or variance)

    if one_hot: label will be parsed to one-hot format
    """
    mean      = tf.constant(config_preprocessing['mean_pixel'], dtype=tf.float32)
    max_value = tf.constant(config_preprocessing['max_pixel'], dtype=tf.float32)
    num_class = config_common['num_classes']
    one_hot = config_preprocessing['one_hot']

    image = tf.divide((tf.cast(image, tf.float32) - mean), max_value)

    if one_hot:
        label = tf.one_hot(tf.cast(label, tf.int64), num_class)
    return image, label


def augmentation_func(image, label):
    with tf.variable_scope("DataAugementation"):
        # random shift
        if config_DataAugmentation['random_shift_switch']:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = _random_rotate(image, config_DataAugmentation['random_rotate_angle'])

        # random color and brightness

        if config_DataAugmentation['random_color_switch']:
            # Random hue`max_hue` must be in the interval [0, 0.5]
            image = tf.image.random_hue(image, max_delta=config_DataAugmentation['max_hue'])
            # Random saturation
            image = tf.image.random_saturation(image,
                                            lower=config_DataAugmentation['lower_sat'],
                                            upper=config_DataAugmentation['upper_sat'])
            # Randm constract
            image = tf.image.random_contrast(image,
                                            lower=config_DataAugmentation['lower_con'],
                                            upper=config_DataAugmentation['upper_con'])
            # Random brightness
            image = tf.image.random_brightness(image, max_delta=config_DataAugmentation['max_bri'])


        # random crop
        if config_DataAugmentation['random_crop_switch']:
            image = _resize_image_shorter_edge(image, config_DataAugmentation['resize_shorter_edge'])
            image = tf.random_crop(image, config_common['input_shape'])
            image = _random_erasing(image, config_DataAugmentation['erasing_max_size'])


        # random noise
        if config_DataAugmentation['random_noise_switch']:
            #image = tf.layers.dropout(image, rate=0.01, training=True)
            image = _gaussian_noise_layer(image, config_DataAugmentation['gaussian_noise_scale'])
            image = _salt_and_pepper_noise(image,
                                           config_DataAugmentation['salt_noise_scale'],
                                           config_DataAugmentation['pepper_noise_scale'])

    return image, label


def resize_output_image(image, label, shape=config_common['input_shape']):
    with tf.variable_scope("ResizeOutputImage"):
        image = tf.image.resize_image_with_crop_or_pad(image, shape[0], shape[1])

    return image, label


def _resize_image_shorter_edge(image, new_shorter_edge=299):
    with tf.variable_scope("resize_image_shorter_edge"):
        shape = tf.shape(image)
        height = shape[0]
        width = shape[1]
        height_smaller_than_width = tf.less_equal(height, width)

        new_height, new_width = control_flow_ops.cond(
            height_smaller_than_width,
            lambda: (new_shorter_edge, (width / height) * new_shorter_edge),
            lambda: (new_shorter_edge, (height / width) * new_shorter_edge))

        new_shape = tf.concat([[new_height], [tf.cast(new_width, tf.int32)]], axis=0)
        
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, new_shape)
        image = tf.squeeze(image, [0])
    return image


def _gaussian_noise_layer(input_tensor, scale):
    with tf.variable_scope("gaussian_noise"):
        noise = tf.random_normal(shape=tf.shape(input_tensor),
                                 mean=0.0, stddev=0.2, dtype=tf.float32)
    return input_tensor + scale*noise


def _salt_and_pepper_noise(input_tensor, salt_ratio, pepper_ratio):
    with tf.variable_scope("SaltAndPepperNoise"):
        random_image = tf.random_uniform(shape=tf.shape(input_tensor),
                                         minval=0.0, maxval=1.0, dtype=tf.float32)
        with tf.variable_scope("salt"):
            salt_image = tf.to_float(tf.greater_equal(
                random_image, 1.0 - salt_ratio))
        with tf.variable_scope("pepper"):
            pepper_image = tf.to_float(
                tf.greater_equal(random_image, pepper_ratio))

        noised_image = tf.minimum(tf.maximum(
            input_tensor, salt_image), pepper_image)

    return noised_image

def _random_rotate(input_tensor, angel):
    with tf.variable_scope("RandomRotate"):
        delta = random_ops.random_uniform([], -angel, angel)
        image = tf.contrib.image.rotate(input_tensor, delta)

    return image

def _random_erasing(image, size):
    with tf.variable_scope("RandomErasing"):
        shape = tf.shape(image)
        w = shape[0]
        h = shape[1]
        shape = tf.constant([1], tf.int32)
        w_ = random_ops.random_uniform_int(shape=shape, minval=0, maxval=w-size)
        h_ = random_ops.random_uniform_int(shape=shape, minval=0, maxval=h-size)
        w_size = random_ops.random_uniform_int(shape=shape, minval=int((size-1)/4), maxval=size-1)
        h_size = random_ops.random_uniform_int(shape=shape, minval=int((size-1)/4), maxval=size-1)

        mask_shape = tf.concat(values=[[w_size], [h_size], [[3]]], axis=1)
        mask_shape = tf.reshape(mask_shape, (-1,))

        uniform_random = random_ops.random_uniform([], 0, 1.0)
        mirror_cond = math_ops.less(uniform_random, .5)
        mask = control_flow_ops.cond(mirror_cond,
            lambda: random_ops.random_normal(shape=mask_shape),
            lambda: tf.zeros(shape=mask_shape))

        paddings = tf.reshape(tf.concat(values=[[[w_, w-w_-w_size]], [[h_, h-h_-h_size]], [[[0], [0]]]], axis=0), (3, 2))

        mask_padding = tf.pad(mask, paddings, mode='CONSTANT', constant_values=1)

    return image*mask_padding

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:52:38 2018

@author: quantummole
"""

import tensorflow as tf

def DoubleConvLayer(inputs,out_channels,kernel_size,padding,training) :
    inputs = tf.layers.conv2d(inputs = inputs,kernel_size=kernel_size,
                     filters = out_channels,activation=None,data_format='channels_first',
                     padding = padding)
    inputs = tf.layers.batch_normalization(inputs,training=training)
    inputs = tf.nn.relu(inputs)
    inputs = tf.layers.conv2d(inputs = inputs,kernel_size=kernel_size,
                     filters = out_channels,activation=None,data_format='channels_first',
                     padding = padding)
    inputs = tf.layers.batch_normalization(inputs,training=training)
    inputs = tf.nn.relu(inputs)
    return inputs    
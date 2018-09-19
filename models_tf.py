# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:14:19 2018

@author: quantummole
"""
import tensorflow as tf
from model_blocks_tf import DoubleConvLayer

#mode = -1 is for debug
#mode = 0 is for test and validation
#mode = {1,2,3..} is for training

class create_net :
    def __init__(self,net) :
        self.net = net
        self.saver = tf.train.Saver(max_to_keep=0)
    def __call__(self,sess,network_params,weights = None) :
        network = self.net(**network_params)
        if weights :
            self.saver.restore(sess,weights)
        return network

def CustomNetClassification(input_dim, final_conv_dim, initial_channels,growth_factor,num_classes) :
    initial_input = tf.placeholder(tf.float32,name = "input",shape=(None,input_dim,input_dim))
    label = tf.placeholder(tf.float32,name = "label",shape=(None,))
    training = tf.placeholder(tf.bool,name = "istraining",shape=())
    inputs = tf.reshape(initial_input,(-1,initial_channels,input_dim,input_dim))
    while input_dim >= final_conv_dim :
        inputs = DoubleConvLayer(inputs,initial_channels+growth_factor,3,"same",training)
        inputs = tf.layers.max_pooling2d(inputs,pool_size = 3,strides= 2,padding="same",data_format="channels_first")
        input_dim = input_dim//2
        initial_channels += growth_factor
    num_units = input_dim*input_dim*initial_channels
    inputs = tf.reshape(inputs,(-1,num_units))
    inputs = tf.layers.dense(inputs=inputs, units=2*num_units, activation=tf.nn.relu)
    inputs = tf.layers.dense(inputs=inputs, units=2*num_units, activation=tf.nn.relu)
    inputs = tf.layers.dense(inputs=inputs, units=num_classes, activation=None)
    return training, [initial_input],[label],[inputs]
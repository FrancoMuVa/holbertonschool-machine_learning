#!/usr/bin/env python3
"""
    Batch Normalization
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    " Creates a batch normalization layer for a neural network in tensorflow "
    epsilon = 1e-7
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    x = tf.keras.layers.Dense(n, kernel_initializer=init, name='layer')(prev)
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    mean, variance = tf.nn.moments(x, axes=[0])
    act = tf.nn.batch_normalization(x, mean, variance,
                                    beta, gamma, epsilon)
    return activation(act)

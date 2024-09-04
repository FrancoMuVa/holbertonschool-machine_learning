#!/usr/bin/env python3
"""
    Layers
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    " Creates a layer "
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    ly = tf.keras.layers.Dense(units=n, name='layer', activation=activation,
                               bias_initializer=init)
    return ly(prev)

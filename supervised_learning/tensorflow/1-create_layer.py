#!/usr/bin/env python3
"""
    Tensorflow
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """ Returns: the tensor output of the layer """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init,
                            name="layer")
    return layer(prev)

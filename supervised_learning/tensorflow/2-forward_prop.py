#!/usr/bin/env python3
"""
    Tensorflow
"""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ creates the forward propagation graph for the neural network """
    prev = x
    for size, activ in zip(layer_sizes, activations):
        prev = create_layer(prev, size, activ)
    return prev

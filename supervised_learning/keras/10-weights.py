#!/usr/bin/env python3
"""
Tensorflow 2 & Keras
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """ Function that saves a model's weights """
    network.save_weights(filepath=filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """ Function that loads a model's weights """
    network.load_weights(filepath=filename)
    return None

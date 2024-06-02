#!/usr/bin/env python3
"""
Tensorflow 2 & Keras
"""
import tensorflow.keras as K


def save_model(network, filename):
    """ Function that saves an entire model """
    network.save(filepath=filename)
    return None


def load_model(filename):
    """ Function that loads an entire model """
    model = K.saving.load_model(filepath=filename)
    return model

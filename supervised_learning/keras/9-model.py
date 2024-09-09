#!/usr/bin/env python3
"""
    Save Only the Best
"""
import tensorflow.keras as K


def save_model(network, filename):
    " saves an entire model "
    network.save(filename)
    return None


def load_model(filename):
    " loads an entire model "
    return K.model.load_model(filename)

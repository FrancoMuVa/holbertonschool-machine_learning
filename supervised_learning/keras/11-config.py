#!/usr/bin/env python3
"""
Tensorflow 2 & Keras
"""
import tensorflow.keras as K


def save_config(network, filename):
    """ Function that saves a model's configuration in JSON format """  
    config = network.to_json()
    with open(filename, 'W') as f:
        f.write(config)
    return None 


def load_config(filename):
    """ Function that loads a model with a specific configuration """
    with open(filename, 'r') as f:
        network = f.read() 
    model = model.from_config(network)
    return model

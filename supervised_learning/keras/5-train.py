#!/usr/bin/env python3
"""
Tensorflow 2 & Keras
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """ Trains a model using mini-batch gradient descent """
    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose,
                       validation_data=validation_data, shuffle=shuffle)

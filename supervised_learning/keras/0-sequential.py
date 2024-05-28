#!/usr/bin/env python3
"""
Tensorflow 2 & Keras
"""
import tensorflow.keras as K
from tensorflow.keras import layers as ly
from tensorflow.keras.regularizers import L2


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Builds a neural network with the Keras library """
    model = K.Sequential()
    model.add(K.Input(shape=(nx,)))
    for i in range(len(layers)):
        model.add(ly.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=L2(lambtha)))
        if i < len(layers) - 1:
            model.add(ly.Dropout(keep_prob))
    return model

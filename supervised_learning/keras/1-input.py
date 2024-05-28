#!/usr/bin/env python3
"""
Tensorflow 2 & Keras
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Builds a neural network with the Keras library """
    input = K.Input(shape=(nx,))

    model_x = K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.L2(lambtha))(input)
    if len(layers) > 1:
        model_x = K.layers.Dropout(1 - keep_prob)(model_x)

    for i in range(1, len(layers)):
        model_x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.L2(lambtha))(model_x)
        if i < len(layers) - 1:
            model_x = K.layers.Dropout(1 - keep_prob)(model_x)

    model = K.Model(inputs=input, outputs=model_x)
    return model

#!/usr/bin/env python3
"""
    Sequential
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    " Builds a neural network with the Keras library "
    input = K.Input(shape=(nx,))
    model = K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha))(input)
    model = K.layers.Dropout(1 - keep_prob)(model)
    for i in range(1, len(layers)):
        model = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha))(model)
        if i < len(layers) - 1:
            model = K.layers.Dropout(1 - keep_prob)(model)
    return K.Model(inputs=input, outputs=model)

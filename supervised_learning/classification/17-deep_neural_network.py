#!/usr/bin/env python3
"""
    Class that define a Deep Neural Network
"""
import numpy as np


class DeepNeuralNetwork:
    """ Class DeepNeuralNetwork """
    def __init__(self, nx, layers):
        " Initializes a new instance of DeepNeuralNetwork "
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(1, len(layers) + 1):
            if not isinstance(layers[i - 1], int) or layers[i - 1] <= 0:
                raise TypeError('layers must be a list of positive integers')
            if i == 1:
                self.__weights['W' + str(i)] = np.random.randn(
                    layers[i - 1], nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W' + str(i)] = np.random.randn(
                    layers[i - 1], layers[i - 2]) * np.sqrt(2 / layers[i - 2])
            self.__weights['b' + str(i)] = np.zeros((layers[i - 1], 1))

    @property
    def L(self):
        " Returns the layers "
        return self.__L

    @property
    def cache(self):
        " Returns the cache "
        return self.__cache

    @property
    def weights(self):
        " Returns the weights "
        return self.__weights
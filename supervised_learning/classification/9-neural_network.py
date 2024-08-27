#!/usr/bin/env python3
"""
    Class neural network with one hidden layer performing binary classification
"""
import numpy as np


class NeuralNetwork:
    """ Class NeuralNetwork """
    def __init__(self, nx, nodes):
        " Initializes NeuralNetwork "
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        elif nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        " Returns W1 "
        return self.__W1

    @property
    def b1(self):
        " Returns b1 "
        return self.__b1

    @property
    def A1(self):
        " Returns A1 "
        return self.__A1

    @property
    def W2(self):
        " Returns W2 "
        return self.__W2

    @property
    def b2(self):
        " Returns b2 "
        return self.__b2

    @property
    def A2(self):
        " Returns A2 "
        return self.__A2

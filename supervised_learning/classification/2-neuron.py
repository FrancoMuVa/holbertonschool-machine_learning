#!/usr/bin/env python3
"""
    Class Neuron that defines a single neuron performing binary classification
"""
import numpy as np


class Neuron:
    " Class Neuron "
    def __init__(self, nx):
        if not isinstance(nx, int):
            " Initializes a new instance of Neuron "
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        " Return the weights attribute "
        return self.__W

    @property
    def b(self):
        " Return the bias attribute "
        return self.__b

    @property
    def A(self):
        " Return the Return the activated output "
        return self.__A

    def forward_prop(self, X):
        " Calculate the forward ropagation "
        x = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-x))
        return self.__A

#!/usr/bin/env python3
"""
    Class Neuron
"""
import numpy as np


class Neuron():
    """ Neuron class """
    def __init__(self, nx):
        """ Initializes a new instance of Neuron """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter method """
        return self.__W

    @property
    def b(self):
        """ Getter method """
        return self.__b

    @property
    def A(self):
        """ Getter method """
        return self.__A

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron """
        x = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-x))
        return self.__A

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost

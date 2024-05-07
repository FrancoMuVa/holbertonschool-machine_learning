#!/usr/bin/env python3
"""
    Class NeuralNetwork.
"""
import numpy as np


class NeuralNetwork():
    """ NeuralNetwork class """

    def __init__(self, nx, nodes):
        """ Initializes a new instance of Neuron """
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
        """ W1 getter """
        return self.__W1

    @property
    def b1(self):
        """ b1 getter """
        return self.__b1

    @property
    def A1(self):
        """ A1 getter """
        return self.__A1

    @property
    def W2(self):
        """ W2 getter """
        return self.__W2

    @property
    def b2(self):
        """ b2 getter """
        return self.__b2

    @property
    def A2(self):
        """ A2 getter """
        return self.__A2

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        x1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-x1))
        x2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-x2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        return -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m

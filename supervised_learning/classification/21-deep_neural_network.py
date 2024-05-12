#!/usr/bin/env python3
"""
    Class DeepNeuralNetwork
"""
import numpy as np


class DeepNeuralNetwork():
    """ DeepNeuralNetwork class """

    def __init__(self, nx, layers):
        """ Initializes a new instance of DeepNeuralNetwork """
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
            if not isinstance(layers[i - 1], int) and layers[i - 1] <= 0:
                raise TypeError('layers must be a list of positive integers')

            if i == 1:
                self.__weights['W' + str(i)] = np.random.randn(
                    layers[i - 1],
                    nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W' + str(i)] = np.random.randn(
                    layers[i - 1],
                    layers[i - 2]) * np.sqrt(2 / layers[i - 2])
            self.__weights['b' + str(i)] = np.zeros((layers[i - 1], 1))

    @property
    def L(self):
        """ Getter method """
        return self.__L

    @property
    def cache(self):
        """ Getter method """
        return self.__cache

    @property
    def weights(self):
        """ Getter method """
        return self.__weights

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            x = np.dot(self.__weights['W' + str(i)],
                       self.__cache['A' + str(i - 1)])
            + self.__weights['b' + str(i)]
            A = 1 / (1 + np.exp(-x))
            self.__cache['A' + str(i)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        return -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m

    def evaluate(self, X, Y):
        """ Evaluates the neural network's predictions """
        self.forward_prop(X)
        A = self.__cache['A' + str(self.__L)]
        pred = (A >= 0.5).astype(int)
        return pred, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network """
        m = Y.shape[1]
        dz = cache['A' + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A = self.__cache['A' + str(i - 1)]
            dw = (1 / m) * np.dot(dz, np.transpose(A))
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            dz = np.dot(np.transpose(self.__weights['W' + str(i)]), dz)\
                * A * (1 - A)
            self.__weights['W' + str(i)] -= alpha * dw
            self.__weights['b' + str(i)] -= alpha * db

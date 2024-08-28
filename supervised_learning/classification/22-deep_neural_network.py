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

    def forward_prop(self, X):
        " Calculates the forward propagation of the neural network "
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            Z = (np.dot(self.__weights['W' + str(i)],
                        self.__cache['A' + str(i - 1)])
                 + self.__weights['b' + str(i)])
            A = 1 / (1 + np.exp(-Z))
            self.__cache['A' + str(i)] = A
        return A, self.__cache

    def cost(self, Y, A):
        " Calculates the cost of the model using logistic regression "
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / \
            Y.shape[1]

    def evaluate(self, X, Y):
        " Evaluates the neural network's predictions "
        _, cahe = self.forward_prop(X)
        A = cahe['A' + str(self.__L)]
        pred = (A >= 0.5).astype(int)
        return pred, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        " Calculates one pass of gradient descent on the neural network "
        m = Y.shape[1]
        dz = cache['A' + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A = cache['A' + str(i - 1)]
            dW = 1 / m * np.matmul(dz, A.T)
            db = 1 / m * np.sum(dz, axis=1, keepdims=True)
            dz = np.matmul(self.__weights['W' + str(i)].T, dz) * A * (1 - A)
            self.__weights['W' + str(i)] -= alpha * dW
            self.__weights['b' + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        " Trains the deep neural network "
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        elif iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        elif alpha < 0:
            raise ValueError('alpha must be positive')
        for _ in range(iterations):
            _, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
        return self.evaluate(X, Y)

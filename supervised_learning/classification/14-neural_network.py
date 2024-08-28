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

    def forward_prop(self, X):
        " Calculates the forward propagation of the neural network "
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        " Calculates the cost of the model using logistic regression "
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / \
            Y.shape[1]

    def evaluate(self, X, Y):
        " Evaluates the neural network's predictions "
        A1, A2 = self.forward_prop(X)
        pred = (A2 >= 0.5).astype(int)
        return pred, self.cost(Y, A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        " Calculates one pass of gradient descent on the neural network "
        m = Y.shape[1]
        Z2 = A2 - Y
        W2 = (1 / m) * np.matmul(Z2, A1.T)
        b2 = (1 / m) * np.sum(Z2, axis=1)

        Z1 = np.matmul(self.__W2.T, Z2) * A1 * (1 - A1)
        W1 = (1 / m) * np.matmul(Z1, X.T)
        b1 = (1 / m) * np.sum(Z1, axis=1, keepdims=True)

        self.__W1 -= alpha * W1
        self.__b1 -= alpha * b1
        self.__W2 -= alpha * W2
        self.__b2 -= alpha * b2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        " Trains the neural network "
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        elif iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        elif alpha < 0:
            raise ValueError('alpha must be positive')
        for _ in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
        return self.evaluate(X, Y)

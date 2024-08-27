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

    def cost(self, Y, A):
        " Calculates the cost of the model using logistic regression "
        return -np.sum(Y * np.log(A) + ((1 - Y) * np.log(1.0000001 - A))) / \
            Y.shape[1]

    def evaluate(self, X, Y):
        " Evaluates the neuron's predictions "
        A = self.forward_prop(X)
        pred = (A >= 0.5).astype(int)
        return pred, self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        " Calculates one pass of gradient descent on the neuron "
        m = X.shape[1]
        a = A - Y
        db = np.sum(a) / m
        dW = np.dot(a, X.T) * 1 / m
        self.__b = self.__b - alpha * db
        self.__W = self.__W - alpha * dW

    def train(self, X, Y, iterations=5000, alpha=0.05):
        " Trains the neuron "
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha=alpha)
        return A, self.cost(Y, A)

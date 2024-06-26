#!/usr/bin/env python3
"""
    Class NeuralNetwork.
"""
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """ Evaluates the neural network's predictions """
        _, A = self.forward_prop(X)
        pred = (A >= 0.5).astype(int)
        return pred, self.cost(Y, A)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network """
        m = Y.shape[1]
        dz2 = A2 - Y
        dW2 = (1 / m) * np.dot(dz2, np.transpose(A1))
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.dot(np.transpose(self.__W2), dz2) * (A1 * (1 - A1))
        dW1 = (1 / m) * np.dot(dz1, np.transpose(X))
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ Trains the neural network """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        elif iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        elif alpha < 0:
            raise ValueError('alpha must be positive')
        if not isinstance(step, int):
            raise TypeError('step must be an integer')
        elif step < 0 or step > iterations:
            raise ValueError('step must be positive and <= iterations')
        x_iter, y_cost = [], []
        for it in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
            cost = self.cost(Y, A2)
            if (it % step == 0):
                if verbose:
                    print(f'Cost after {it} iterations: {cost}')
                if graph:
                    x_iter.append(it)
                    y_cost.append(cost)
        plt.plot(x_iter, y_cost, '-', c='blue')
        plt.title('Training Cost')
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.show()
        return self.evaluate(X, Y)

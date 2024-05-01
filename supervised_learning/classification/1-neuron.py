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

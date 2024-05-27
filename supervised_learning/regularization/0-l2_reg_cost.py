#!/usr/bin/env python3
"""
    Regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Calculates the cost of a neural network with L2 regularization """
    sum_weights = 0
    for i in range(L):
        W = weights['W' + str(i + 1)]
        sum_weights += np.sum(np.square(W))

    return sum_weights * (lambtha / (2 * m)) + cost

#!/usr/bin/env python3
"""
    L2 Regularization Cost
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    " Calculates the cost of a neural network with L2 regularization "
    sum = 0
    for i in range(L):
        w = weights['W' + str(i + 1)]
        sum += np.sum(np.square(w))
    return sum * (lambtha / (2 * m)) + cost

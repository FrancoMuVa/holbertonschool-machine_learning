#!/usr/bin/env python3
"""
    Moving Average
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
        Updates a variable using the gradient descent with momentum
        optimization algorithm.
    """
    v = beta1 * v + (1 - beta1) * grad
    return var - alpha * v, v

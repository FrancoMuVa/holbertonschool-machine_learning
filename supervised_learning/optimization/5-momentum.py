#!/usr/bin/env python3
"""
    Optimization
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
        Updates a variable using the gradient descent with momentum
        optimization algorithm
    """
    v1 = beta1 * v + (1 - beta1) * grad
    var1 = var - alpha * v1
    return var1, v1

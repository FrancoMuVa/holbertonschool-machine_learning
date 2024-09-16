#!/usr/bin/env python3
"""
    RMSProp
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    " Updates a variable in place using the Adam optimization algorithm "
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    bc_v = v / (1 - beta1 ** t)
    bc_s = s / (1 - beta2 ** t)
    var = var - alpha * bc_v / (np.sqrt(bc_s) + epsilon)
    return var, v, s

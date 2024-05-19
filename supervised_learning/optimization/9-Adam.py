#!/usr/bin/env python3
"""
    Optimization
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ Updates a variable in place using the Adam optimization algorithm """
    v1 = beta1 * v + (1 - beta1) * grad
    s1 = beta2 * s + (1 - beta2) * (grad ** 2)

    bias_correction_v = v1 / (1 - beta1 ** t)
    bias_correction_s = s1 / (1 - beta2 ** t)
    var1 = var - alpha * bias_correction_v / (np.sqrt(
        bias_correction_s) + epsilon)
    return var1, v, s

#!/usr/bin/env python3
"""
    Optimization
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ Updates a variable using the RMSProp optimization algorithm """
    s1 = beta2 * s + (1 - beta2) * (grad ** 2)
    var1 = var - alpha * grad / (np.sqrt(s1) + epsilon)
    return var1, s1

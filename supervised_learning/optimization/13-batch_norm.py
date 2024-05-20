#!/usr/bin/env python3
"""
    Optimization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
        Normalizes an unactivated output of a
        neural network using batch normalization
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_n = (Z - mean) / np.sqrt(var + epsilon)
    return Z_n * gamma + beta

#!/usr/bin/env python3
"""
    Batch Normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
        Normalizes an unactivated output of a neural
        network using batch normalization.
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    z = (Z - mean) / (np.sqrt((var ** 2) + epsilon))
    return z * gamma + beta

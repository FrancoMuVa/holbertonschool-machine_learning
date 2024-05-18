#!/usr/bin/env python3
"""
    Optimization
"""
import numpy as np


def shuffle_data(X, Y):
    """ shuffles the data points in two matrices the same way """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    print(f'--->>> {permutation}')
    return X[permutation], Y[permutation]

#!/usr/bin/env python3
"""
    Shuffle Data
"""
import numpy as np


def shuffle_data(X, Y):
    " shuffles the data points in two matrices the same way "
    m = X.shape[0]
    p = np.random.permutation(m)
    return X[p], Y[p]

#!/usr/bin/env python3
"""
    Mini-Batch
"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
        Creates mini-batches to be used for training a neural network using
        mini-batch gradient descent
    """
    shuffle_X, shuffle_Y = shuffle_data(X, Y)
    mini_batches = []
    m = X.shape[0]
    for i in range(0, m, batch_size):
        X_batch = shuffle_X[i:i + batch_size]
        Y_batch = shuffle_Y[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches

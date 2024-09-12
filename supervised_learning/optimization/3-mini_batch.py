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
    q_batches = m // batch_size
    sta, end = 0, batch_size
    for i in range(q_batches):
        sta = i * batch_size
        end = (i + 1) * batch_size
        X_batch = shuffle_X[sta:end]
        Y_batch = shuffle_Y[sta:end]
        mini_batches.append((X_batch, Y_batch))
    if m % batch_size != 0:
        r = m % batch_size
        X_batch = shuffle_X[sta:r]
        Y_batch = shuffle_Y[sta:r]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches

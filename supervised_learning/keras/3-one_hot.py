#!/usr/bin/env python3
"""
Tensorflow 2 & Keras
"""
import numpy as np


def one_hot(labels, classes=None):
    """ Converts a label vector into a one-hot matrix """
    if classes is None:
        classes = np.max(labels) + 1
    matrix = np.zeros((len(labels), classes))
    matrix[np.arange(len(labels)), labels] = 1
    return matrix

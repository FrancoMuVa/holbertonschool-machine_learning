#!/usr/bin/env python3
"""
    Sensitivity
"""
import numpy as np


def sensitivity(confusion):
    " Calculates the sensitivity for each class in a confusion matrix "
    matrix = np.zeros((confusion.shape[0]))
    diag = np.diag(confusion)
    tp_fn = np.sum(confusion, axis=1)
    matrix = (diag / tp_fn)
    return matrix

#!/usr/bin/env python3
"""
    Precision
"""
import numpy as np


def precision(confusion):
    " Calculates the precision for each class in a confusion matrix "
    matrix = np.zeros((confusion.shape[0]))
    true_positives = np.diag(confusion)
    retrieved_instances = np.sum(confusion, axis=0)
    matrix = (true_positives / retrieved_instances)
    return matrix

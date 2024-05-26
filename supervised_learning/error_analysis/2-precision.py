#!/usr/bin/env python3
"""
    Error analysis
"""
import numpy as np


def precision(confusion):
    """ calculates the precision for each class in a confusion matrix """
    TP = np.diagonal(confusion)
    FN = np.sum(confusion, axis=0) - TP
    return TP / (TP + FN)

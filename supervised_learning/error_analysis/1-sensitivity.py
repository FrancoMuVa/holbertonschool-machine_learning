#!/usr/bin/env python3
"""
    Error analysis
"""
import numpy as np


def sensitivity(confusion):
    """ calculates the sensitivity for each class in a confusion matrix """
    TP = np.diagonal(confusion)
    FN = np.sum(confusion, axis=1) - TP
    return TP / (TP + FN)

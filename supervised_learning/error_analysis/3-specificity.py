#!/usr/bin/env python3
"""
    Error analysis
"""
import numpy as np


def specificity(confusion):
    """ Calculates the specificity for each class in a confusion matrix """
    TP = np.diagonal(confusion)
    FN = np.sum(confusion, axis=1) - TP
    FP = np.sum(confusion, axis=0) - TP
    TN = np.sum(confusion) - (TP + FN + FP)
    return TN / (TN + FP)

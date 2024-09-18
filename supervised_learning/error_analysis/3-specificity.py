#!/usr/bin/env python3
"""
    Specificity
"""
import numpy as np


def specificity(confusion):
    " Calculates the specificity for each class in a confusion matrix "
    matrix = np.zeros((confusion.shape[0]))
    tp = np.diag(confusion)
    fn = np.sum(confusion, axis=1) - tp
    fp = np.sum(confusion, axis=0) - tp
    tn = np.sum(confusion)
    tn -= (tp + fn + fp)
    matrix = tn / (tn + fp)
    return matrix

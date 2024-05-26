#!/usr/bin/env python3
"""
    Error analysis
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ Calculates the F1 score of a confusion matrix """
    P = precision(confusion)
    S = sensitivity(confusion)
    return (2 * (P * S) / (P + S))

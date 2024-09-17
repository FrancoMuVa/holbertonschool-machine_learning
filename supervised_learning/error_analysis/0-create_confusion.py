#!/usr/bin/env python3
"""
    Create Confusion
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    " Creates a confusion matrix "
    matrix = np.zeros((labels.shape[1], labels.shape[1]))
    correct_idx = np.argmax(labels, axis=1)
    pred_idx = np.argmax(logits, axis=1)
    for c, p in zip(correct_idx, pred_idx):
        matrix[c, p] += 1
    return matrix

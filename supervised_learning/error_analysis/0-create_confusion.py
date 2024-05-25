#!/usr/bin/env python3
"""
    Error analysis
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """ Creates a confusion matrix """
    classes = labels.shape[1]

    conf_matrix = np.zeros((classes, classes), dtype=float)
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    for true, pred in zip(true_classes, pred_classes):
        conf_matrix[true, pred] += 1
    return conf_matrix

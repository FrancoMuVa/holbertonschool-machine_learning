#!/usr/bin/env python3
"""
    Accuracy
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    " Calculates the accuracy of a prediction "
    idx_data = tf.argmax(y, 1)
    idx_pred_classes = tf.argmax(y_pred, 1)
    bool_tensor = tf.math.equal(idx_data, idx_pred_classes)
    accuracy = tf.reduce_mean(tf.cast(bool_tensor, dtype=tf.float32))
    return accuracy

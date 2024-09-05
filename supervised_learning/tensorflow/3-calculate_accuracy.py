#!/usr/bin/env python3
"""
    Accuracy
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    " Calculates the accuracy of a prediction "
    bool_tensor = tf.math.equal(y, y_pred)
    accuracy = tf.reduce_mean(tf.cast(bool_tensor, dtype=tf.float32))
    return accuracy

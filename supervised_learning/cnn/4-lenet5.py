#!/usr/bin/env python3
"""
LeNet-5 (Tensorflow 1)
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    " Builds a modified version of the LeNet-5 architecture using tensorflow "
    he_normal = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv1 = tf.layers.Conv2D(filters=6,
                             kernel_size=5,
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=he_normal)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv1)

    conv2 = tf.layers.Conv2D(filters=16,
                             kernel_size=5,
                             padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=he_normal)(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)

    flat1 = tf.layers.Flatten()(pool2)
    fully1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                             kernel_initializer=he_normal)(flat1)
    fully2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                             kernel_initializer=he_normal)(fully1)
    logits = tf.layers.Dense(units=10, activation=tf.nn.relu)(fully2)

    y_pred = tf.nn.softmax(logits)
    pre_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                          logits=logits)
    loss = tf.reduce_mean(pre_loss)
    op = tf.train.AdamOptimizer()
    train_op = op.minimize(loss)
    corr_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(corr_pred, dtype=tf.float32))
    return y_pred, train_op, loss, acc

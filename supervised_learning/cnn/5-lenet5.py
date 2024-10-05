#!/usr/bin/env python3
"""
LeNet-5 (Keras)
"""
from tensorflow import keras as K


def lenet5(X):
    " Builds a modified version of the LeNet-5 architecture using keras "
    he_normal = K.initializers.he_normal(seed=0)

    conv1 = K.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                            activation='relu',
                            kernel_initializer=he_normal)(X)
    pool1 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                            activation='relu',
                            kernel_initializer=he_normal)(pool1)
    pool2 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flatten = K.layers.Flatten()(pool2)
    fully_1 = K.layers.Dense(units=120, kernel_initializer=he_normal,
                             activation='relu')(flatten)
    fully_2 = K.layers.Dense(units=84, kernel_initializer=he_normal,
                             activation='relu')(fully_1)
    logits = K.layers.Dense(units=10, activation='softmax',
                            kernel_initializer=he_normal)(fully_2)

    model = K.models.Model(inputs=X, outputs=logits)
    adam = K.optimizers.Adam()
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

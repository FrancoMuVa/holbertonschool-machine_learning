#!/usr/bin/env python3
"""
    Learning Rate Decay
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent,
    analyze validation data and train the model with learning rate decay.
    """
    callback = []
    if learning_rate_decay and validation_data:
        callback.append(K.callbacks.LearningRateScheduler(
            lambda epochs: alpha / (1 + (decay_rate * epochs)), verbose=True))
    if early_stopping:
        callback.append(K.callbacks.EarlyStopping('val_loss',
                                                  patience=patience))
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callback,
        validation_data=validation_data,
        shuffle=shuffle)

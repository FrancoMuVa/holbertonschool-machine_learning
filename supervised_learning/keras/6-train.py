#!/usr/bin/env python3
"""
    Early Stopping
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent
    and analyze validation data.
    """
    if early_stopping:
        callback = K.callbacks.EarlyStopping('val_loss', patience=patience)
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=[callback],
        validation_data=validation_data,
        shuffle=shuffle)

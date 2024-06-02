#!/usr/bin/env python3
"""
Tensorflow 2 & Keras
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None, verbose=True,
                shuffle=False):
    """ Trains a model using mini-batch gradient descent """
    callbacks = []
    if learning_rate_decay and validation_data:
        callbacks.append(K.callbacks.LearningRateScheduler(
            lambda epoch: alpha / (1 + decay_rate * epoch), verbose=True))

    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=patience))
    if save_best:
        best = K.callbacks.ModelCheckpoint(filepath=filepath,
                                           monitor="val_loss")
        callbacks.append(best)
    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, callbacks=callbacks,
                       validation_data=validation_data, shuffle=shuffle)

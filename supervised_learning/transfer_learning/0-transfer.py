#!/usr/bin/env python3
"""
0. Transfer Knowledge
"""
import tensorflow.keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    Function that pre-processes the data.
    """
    x = tf.image.resize(X, (71, 71))
    x = K.applications.xception.preprocess_input(x)
    y = K.utils.to_categorical(Y, 10)
    return x, y


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    xception = K.applications.xception.Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(71, 71, 3)
    )

    xception.trainable = False

    inputs = K.Input(shape=(71, 71, 3), name='input')

    inputs = K.layers.Lambda(
        lambda x: tf.image.resize(x, (71, 71)))(inputs)

    x = xception(inputs)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Flatten(name='Flatten')(x)

    x = K.layers.Dense(
        512,
        activation='relu',
        kernel_regularizer=K.regularizers.L2(0.0001),
        name='Dense_1'
    )(x)
    x = K.layers.Dropout(0.3, name='Dropout_1')(x)
    x = K.layers.BatchNormalization()(x)

    outputs = K.layers.Dense(10, activation='softmax', name='output')(x)

    model = K.Model(inputs=inputs, outputs=outputs, name='model')

    # model.summary()

    datagen = K.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True
    )

    adam = K.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = K.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )

    lr_scheduler = K.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    model.fit(
        datagen.flow(x_train, y_train, batch_size=100),
        validation_data=(x_test, y_test),
        epochs=35,
        verbose=True,
        callbacks=[early_stopping, lr_scheduler]
    )

    model.save('cifar10.h5')

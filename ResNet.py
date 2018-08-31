# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

from util import plot

CLASSES = 10
ROWS, COLS, CHS = 32, 32, 3

TRAIN_SIZE = 50000
TEST_SIZE = 10000
VALIDATION_SPLIT = TEST_SIZE / TRAIN_SIZE

BATCH_SIZE = 100
EPOCHS = 16

def prepare():
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    Y_train = to_categorical(Y_train, CLASSES)
    Y_test = to_categorical(Y_test, CLASSES)

    return (X_train, Y_train), (X_test, Y_test)


def residual(inputs, filters, kernel_size, down_sampling=False):
    input_shape = K.shape(inputs)
    output_shape = input_shape

    if down_sampling:
        outputs = keras.layers.Conv2D(filters, kernel_size, strides=(2, 2), padding='same')(inputs)
    else:
        outputs = keras.layers.Conv2D(filters, kernel_size, padding='same')(inputs)

    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.Activation(keras.activations.relu)(outputs)
    outputs = keras.layers.Conv2D(filters, kernel_size, padding='same')(outputs)

    if down_sampling:
        inputs = keras.layers.Conv2D(filters, kernel_size, strides=(2, 2), padding='same')(inputs)
    # elif input_shape[-1] != filters:
    #     inputs = keras.layers.Conv2D(filters, (1, 1), padding='same')(inputs)

    outputs = keras.layers.Add()([inputs, outputs])
    outputs = keras.layers.Activation(keras.activations.relu)(outputs)

    return outputs


def build(input_shape, n=3):
    """

    :param input_shape:
    :param n: Variant for ResNet for Cifar10 (n=3 => 20, n=5 => 32, n=7 => 44, n=9 => 56 layer)
    :return:
    """
    inputs = keras.Input(shape=input_shape)

    outputs = keras.layers.Conv2D(16, (3, 3), padding='same')(inputs)

    for i in range(n):
        outputs = residual(outputs, 16, (3, 3))

    for i in range(n):
        if i == 0:
            outputs = residual(outputs, 32, (3, 3), down_sampling='same')
        else:
            outputs = residual(outputs, 32, (3, 3))

    for i in range(n):
        if i == 0:
            outputs = residual(outputs, 64, (3, 3), down_sampling='same')
        else:
            outputs = residual(outputs, 64, (3, 3))

    outputs = keras.layers.GlobalAveragePooling2D()(outputs)
    outputs = keras.layers.Dense(CLASSES)(outputs)
    outputs = keras.layers.Activation(keras.activations.relu)(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    # To plot model, commnet out below line.
    # keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    return model


if __name__ == '__main__':

    (X_train, Y_train), (X_test, Y_test) = prepare()

    model = build(input_shape=(ROWS, COLS, CHS, ))

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    history = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=2,
                        validation_split=VALIDATION_SPLIT)

    plot(history, metrics=['val', 'acc'])

    score = model.evaluate(X_test, Y_test)

    print("loss: ", score[0])
    print("acc:  ", score[1])

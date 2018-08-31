# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

CLASSES = 10
ROWS, COLS, CHS = 32, 32, 3


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
    elif input_shape[-1] != filters:
        inputs = keras.layers.Conv2D(filters, (1, 1), padding='same')(inputs)
    outputs = keras.layers.Add()([inputs, outputs])
    # outputs = inputs + outputs
    outputs = keras.layers.Activation(keras.activations.relu)(outputs)
    return outputs


def build(input_shape, n=3):
    """

    :param input_shape:
    :param n: Variant for ResNet for Cifar10 (n=3 => 20, n=
    :return:
    """
    inputs = keras.Input(shape=input_shape)

    outputs = keras.layers.Conv2D(16, (3, 3), padding='same')(inputs)

    # outputs = residual(outputs, 16, (3, 3), down_sampling='same')
    outputs = residual(outputs, 16, (3, 3))
    outputs = residual(outputs, 16, (3, 3))
    outputs = residual(outputs, 16, (3, 3))

    outputs = residual(outputs, 32, (3, 3), down_sampling='same')
    outputs = residual(outputs, 32, (3, 3))
    outputs = residual(outputs, 32, (3, 3))

    outputs = residual(outputs, 64, (3, 3), down_sampling='same')
    outputs = residual(outputs, 64, (3, 3))
    outputs = residual(outputs, 64, (3, 3))

    outputs = keras.layers.GlobalAveragePooling2D()(outputs)
    outputs = keras.layers.Dense(CLASSES)(outputs)
    outputs = keras.layers.Activation(keras.activations.relu)(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    return model


if __name__ == '__main__':

    (X_train, Y_train), (X_test, Y_test) = prepare()

    # inputs = keras.Input(shape=(ROWS, COLS, CHS, ))
    # outputs = residual(inputs, 32, (3, 3), down_sampling=True)
    # outputs = residual(outputs, 32, (3, 3))
    # outputs = Flatten()(outputs)
    # outputs = Dense(1024, activation='relu')(outputs)
    # outputs = Dense(CLASSES, activation='softmax') (outputs)
    # model = keras.Model(inputs=inputs, outputs=outputs)
    #
    # model.summary()

    model = build(input_shape=(ROWS, COLS, CHS, ))

    # model = Sequential([
    #     BatchNormalization(axis=-1, input_shape=(ROWS, COLS, CHS, )),
    #     Residual(16, (3, 3)),
    #     Residual(32, (3, 3)),
    #     Dense(300, activation='relu'),
    #     Dense(100, activation='relu'),
    #     Dense(CLASSES, activation='softmax')
    # ])

    # model.summary()

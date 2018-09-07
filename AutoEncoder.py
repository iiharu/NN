# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot
from tensorflow import keras

from datasets import mnist
from utils import plot

from layers import conv2d, cropping2d, dense, flatten, transposed_conv2d, zero_padding2d

CLASSES = 10
ROWS, COLS, CHS = 28, 28, 1
TRAIN_SIZE = 60000
TEST_SIZE = 10000

BATCH_SIZE = 100
EPOCHS = 8
VALIDATION_SPLIT = TEST_SIZE / TRAIN_SIZE


def build(input_shape):

    inputs = keras.layers.Input(shape=input_shape)

    outputs = zero_padding2d(padding=2)(inputs)
    outputs = conv2d(4, (3, 3), strides=2)(outputs)
    outputs = transposed_conv2d(1, (3, 3), strides=2)(outputs)
    outputs = cropping2d((2, 2))(outputs)

    outputs = flatten()(outputs)
    outputs = dense(300, activation=keras.activations.relu)(outputs)
    outputs = dense(100, activation=keras.activations.relu)(outputs)
    outputs = dense(CLASSES, activation=keras.activations.softmax)(outputs)

    model = keras.Model(inputs, outputs)

    model.summary()

    return model


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data(normalize=True,
                                                           one_hot_label=True)

    model = build(input_shape=(ROWS, COLS, CHS, ))

    model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=2,
                        validation_split=VALIDATION_SPLIT)

    score = model.evaluate(X_test, Y_test, verbose=0)

    print(score)

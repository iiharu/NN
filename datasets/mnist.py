# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

CLASSES = 10
ROWS, COLS, CHS = 28, 28, 1

TRAIN_SIZE = 50000
TEST_SIZE = 10000

def load_data(normalize=False, flatten=False, one_hot_label=True):
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    if normalize:
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        X_train /= 255
        Y_test /= 255

    if flatten:
        X_train = X_train.reshape((TRAIN_SIZE, ROWS * COLS * CHS))
        X_test = X_test.reshape((TEST_SIZE, ROWS * COLS * CHS))

    if one_hot_label:
        Y_train = to_categorical(Y_train, CLASSES)
        Y_test = to_categorical(Y_test, CLASSES)

    return (X_train, Y_train), (X_test, Y_test)

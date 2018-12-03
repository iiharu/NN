# -*- coding: utf-8 -*-

import numpy as np
from tensorflow import keras

from models import LeNet5

from utils import plot
from datasets import mnist

from layers import dense, flatten
from tensorflow.keras.layers import BatchNormalization

CLASSES = 10
ROWS, COLS, CHS = 28, 28, 1

TRAIN_SIZE = 50000
TEST_SIZE = 10000
VALIDATION_SPLIT = TEST_SIZE / TRAIN_SIZE

BATCH_SIZE = 128
EPOCHS = 16


def build(input_shape, classes=CLASSES):
    model = keras.Sequential([
        flatten(input_shape=input_shape),
        BatchNormalization(center=False, scale=False),
        dense(units=classes)
    ])
    model.summary()

    return model


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    model = build(input_shape=(ROWS, COLS, CHS, ), classes=CLASSES)
    # model = LeNet5().build(input_shape=(ROWS, COLS, CHS, ), classes=CLASSES)

    #keras.utils.plot_model(model, to_file='model.png')

    # model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
    #              loss=keras.losses.categorical_crossentropy,
    #              metrics=['acc'])

    # history = model.fit(X_train, Y_train,
    #                    batch_size=BATCH_SIZE,
    #                    epochs=EPOCHS,
    #                    validation_data=(X_test, Y_test),
    #                    verbose=2)

    #plot(history, metrics=['loss', 'acc'])

    #score = model.evaluate(X_test, Y_test, verbose=0)

    #print("loss: ", score[0])
    #print("acc:  ", score[1])

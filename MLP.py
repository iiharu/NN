# -*- coding: utf-8 -*-

"""
MLP.py

DataSet: MNIST
Model: LeNet-300-100

"""

import numpy as np
from tensorflow import keras

from util import plot

CLASSES = 10
ROWS, COLS = 28, 28
TRAIN_SIZE = 60000
TEST_SIZE = 10000

BATCH_SIZE = 100
EPOCHS = 8
VALIDATION_SPLIT = TEST_SIZE / TRAIN_SIZE

# class LeNet(keras.Model):

#     def __init__(self, input_shape=(784, ), classes=10):
#         super(LeNet, self).__init__(name='LeNet')

#         self.batch_normalize = karas.layers.BatchNormalization(axis=-1, input_shape=input_shape)
#         self.dense1 = keras.layers.Dense(300, activation=keras.activations.relu)
#         self.dense2 = Dense(100, activation=keras.activations.relu)
#         self.dense3 = Dense(CLASSES, activation=keras.activations.softmax)

#     def call(self, inputs):
#         x = self.batch_normalize(x)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         return self.dense3(x)


def prepare():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    X_train = np.reshape(X_train, (TRAIN_SIZE, ROWS * COLS))
    X_test =  np.reshape(X_test, (TEST_SIZE, ROWS * COLS))
    Y_train = keras.utils.to_categorical(Y_train, CLASSES)
    Y_test = keras.utils.to_categorical(Y_test, CLASSES)

    return (X_train, Y_train), (X_test, Y_test)

def build():

    model = keras.Sequential([
        keras.layers.BatchNormalization(axis=-1, input_shape=(ROWS * COLS, )),
        keras.layers.Dense(300, activation=keras.activations.relu),
        keras.layers.Dense(100, activation=keras.activations.relu),
        keras.layers.Dense(CLASSES, activation=keras.activations.softmax)
    ])

    model.summary()
    
    return model

if __name__ == '__main__':

    print("keras", keras.__version__)
    
    (X_train, Y_train), (X_test, Y_test) = prepare()

    model = build()

    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    history = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=2,
                        validation_split=VALIDATION_SPLIT)

    plot(history, metrics=['loss', 'acc'])
    
    score = model.evaluate(X_test, Y_test)

    print("loss: ", score[0])
    print("acc: ", score[1])

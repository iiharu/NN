# -*- coding: utf-8 -*-

import numpy as np
from tensorflow import keras

CLASSES = 10
ROWS, COLS = 28, 28
TRAIN_SIZE = 60000
TEST_SIZE = 10000

BATCH_SIZE = 100
EPOCHS = 8
VALIDATION_SPLIT = TEST_SIZE / TRAIN_SIZE

def prepare():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    X_train = np.reshape(X_train, (TRAIN_SIZE, ROWS * COLS))
    X_test =  np.reshape(X_test, (TEST_SIZE, ROWS * COLS))
    Y_train = keras.utils.to_categorical(Y_train, CLASSES)
    Y_test = keras.utils.to_categorical(Y_test, CLASSES)

    return (X_train, Y_train), (X_test, Y_test)

def build():
    model = keras.Sequential()
    model.add(keras.layers.BatchNormalization(axis=-1, input_shape=(ROWS * COLS, )))
    model.add(keras.layers.Dense(300))
    model.add(keras.layers.Activation(keras.activations.relu))
    model.add(keras.layers.Dense(100))
    model.add(keras.layers.Activation(keras.activations.relu))
    model.add(keras.layers.Dense(CLASSES))
    model.add(keras.layers.Activation(keras.activations.softmax))

    model.summary()
    
    return model

if __name__ == '__main__':

    print("keras", keras.__version__)
    
    (X_train, Y_train), (X_test, Y_test) = prepare()

    model = build()

    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    model.fit(X_train, Y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_split=VALIDATION_SPLIT)

    score = model.evaluate(X_test, Y_test)

    print("loss: ", score[0])
    print("acc: ", score[1])

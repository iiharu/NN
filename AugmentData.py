# -*- coding: utf-8 -*-

import numpy as np
from tensorflow import keras

from utils import plot

CLASSES = 10
ROWS, COLS, CHS = 32, 32, 3

BATCH_SIZE = 500
EPOCHS = 16

TRAIN_SIZE = None
TEST_SIZE = None
VALIDATION_SPLIT = None
INPUT_SHAPE = None


def prepare():

    global TRAIN_SIZE
    global TEST_SIZE
    global INPUT_SHAPE
    global VALIDATION_SPLIT
    
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

    TRAIN_SIZE = len(X_train)
    TEST_SIZE = len(X_test)
    VALIDATION_SPLIT = TEST_SIZE / TRAIN_SIZE
    if keras.backend.image_data_format() == 'channels_last':
        X_train = np.reshape(X_train, (TRAIN_SIZE, ROWS, COLS, CHS))
        X_test = np.reshape(X_test, (TEST_SIZE, ROWS, COLS, CHS))
        INPUT_SHAPE = (ROWS, COLS, CHS, )
    else:
        X_train = np.reshape(X_train, (len(X_train), CHS, ROWS, COLS))
        X_test = np.reshape(X_test, (len(X_test), CHS, ROWS, COLS))
        INPUT_SHAPE = (CHS, ROWS, COLS, )
    Y_train = keras.utils.to_categorical(Y_train, CLASSES)
    Y_test = keras.utils.to_categorical(Y_test, CLASSES)

    return (X_train, Y_train), (X_test, Y_test)


def build():

    model = keras.Sequential([
        keras.layers.BatchNormalization(axis=-1, input_shape=INPUT_SHAPE),
        # keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='same'),
        keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1)),
        # keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Activation(keras.activations.sigmoid),
        # keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same'),
        keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1)),
        # keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Activation(keras.activations.sigmoid),
        keras.layers.Flatten(),
        keras.layers.Dense(120),
        keras.layers.Dense(64),
        keras.layers.Dense(CLASSES),
        keras.layers.Activation(keras.activations.softmax)
    ])

    model.summary()
  
    return model


if __name__ == '__main__':

    (X_train, Y_train), (X_test, Y_test) = prepare()

    model = build()

    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    # Data Augumentation
    datagen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
                                                           featurewise_std_normalization=True,
                                                           rotation_range=20,
                                                           width_shift_range=0.2,
                                                           height_shift_range=0.2,
                                                           horizontal_flip=True)
    datagen.fit(X_train)

    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                  steps_per_epoch=TRAIN_SIZE,
                                  epochs=EPOCHS,
                                  verbose=2)

    plot(history)
    
    score = model.evaluate(X_test, Y_test)

    print("loss: ", score[0])
    print("acc: ", score[1])

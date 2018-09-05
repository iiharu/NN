# -*- coding: utf-8 -*-

import numpy as np
from tensorflow import keras

from layers import concat, batch_normalization, relu, average_pooling2d, global_average_pooling2d, softmax
from utils import plot

CLASSES = 10
ROWS, COLS, CHS = 32, 32, 3

TRAIN_SIZE = 50000
TEST_SIZE = 10000
VALIDATION_SPLIT = TEST_SIZE / TRAIN_SIZE

GROWTH_RATE = 12
LAYERS = 40
BLOCKS = 3
BATCH_SIZE = 128
EPOCHS = 32


def prepare():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
    Y_train = keras.utils.to_categorical(Y_train, CLASSES)
    Y_test = keras.utils.to_categorical(Y_test, CLASSES)

    return (X_train, Y_train), (X_test, Y_test)


def conv2d(filters, kernel_size, strides=1):
    return keras.layers.Conv2D(filters,
                               kernel_size,
                               strides=strides,
                               padding='same',
                               kernel_initializer=keras.initializers.he_normal(),
                               kernel_regularizer=keras.regularizers.l2(0.0001),
                               use_bias=False)


def dense(units):
    return keras.layers.Dense(units,
                              kernel_regularizer=keras.regularizers.l2(0.0001))


def dense_conv(inputs, growth_rate, layers, kernel_size=(3, 3)):

    outputs = []

    for k in range(layers):
        if k <= 1:
            inputs = batch_normalization()(inputs)
            inputs = relu()(inputs)
            inputs = conv2d(growth_rate, kernel_size)(inputs)
        else:
            inputs = concat()(outputs)
            inputs = batch_normalization()(inputs)
            inputs = relu()(inputs)
            inputs = conv2d(growth_rate, kernel_size)(inputs)
        outputs.append(inputs)

    outputs = concat()(outputs)

    return outputs


def transition(inputs, filters=12):
    outputs = conv2d(filters, kernel_size=(1, 1))(inputs)
    outputs = average_pooling2d(pool_size=(2, 2), strides=2)(outputs)
    return outputs

def build(input_shape):

    k = GROWTH_RATE
    n = BLOCKS
    l = (LAYERS - 4) // 3 // 3

    inputs = keras.Input(shape=input_shape)

    outputs = conv2d(16, kernel_size=(3, 3))(inputs)

    for i in range(n):
        outputs = dense_conv(outputs, growth_rate=k, layers=l)
        outputs = transition(outputs, k * l)
        k += GROWTH_RATE

    outputs = global_average_pooling2d()(outputs)
    outputs = dense(CLASSES)(outputs)
    outputs = softmax()(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    # To plot model, commnet out below line.
    # keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    return model


if __name__ == '__main__':

    (X_train, Y_train), (X_test, Y_test) = prepare()

    X_train, X_val = np.split(X_train, [45000], axis=0)
    Y_train, Y_val = np.split(Y_train, [45000], axis=0)

    model = build(input_shape=(ROWS, COLS, CHS, ))

    model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    datagen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=4,
                                                           height_shift_range=4,
                                                           horizontal_flip=True)

    datagen.fit(X_train)

    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                  steps_per_epoch=TRAIN_SIZE // 10,
                                  epochs=EPOCHS,
                                  verbose=2,
                                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, mode='auto')],
                                  validation_data=(X_val, Y_val))

    plot(history, metrics=['loss', 'acc'])

    score = model.evaluate(X_test, Y_test, verbose=0)

    print("loss: ", score[0])
    print("acc:  ", score[1])

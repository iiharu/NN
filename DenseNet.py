# -*- coding: utf-8 -*-

import numpy as np
from tensorflow import keras
from tensorflow.python.keras import backend as K

from layers import concat, batch_normalization, relu, average_pooling2d, global_average_pooling2d, softmax
from utils import plot

CLASSES = 10
ROWS, COLS, CHS = 32, 32, 3

TRAIN_SIZE = 50000
TEST_SIZE = 10000
VALIDATION_SPLIT = TEST_SIZE / TRAIN_SIZE

# GROWTH_RATE = 12
GROWTH_RATE = 16
LAYERS = 100
BLOCKS = 3
BATCH_SIZE = 64
EPOCHS = 32

BN_AXIS = 3 if K.image_data_format() == 'channels_last' else 1


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


def conv_block(inputs, growth_rate):
    outputs = batch_normalization()(inputs)
    outputs = relu()(outputs)
    outputs = conv2d(filters=4 * growth_rate, kernel_size=(1, 1))(outputs)
    outputs = batch_normalization()(outputs)
    outputs = relu()(outputs)
    outputs = conv2d(filters=growth_rate, kernel_size=(3, 3))(outputs)
    outputs = concat()([inputs, outputs])

    return outputs


def dense_block(inputs, blocks):
    for i in range(blocks):
        inputs = conv_block(inputs, GROWTH_RATE)
    return inputs


# def dense_conv_block(inputs, filters, layers, growth_rate=GROWTH_RATE):
#     for k in range(layers):
#         outputs = batch_normalization()(inputs)
#         outputs = relu()(outputs)
#         outputs = conv2d(filters=128, kernel_size=(1, 1))(outputs) # filters mean 4 * growth_rate?
#         outputs = batch_normalization()(outputs)
#         outputs = relu()(outputs)
#         outputs = conv2d(filters=growth_rate, kernel_size=(3, 3))(outputs)
#         outputs = concat()([inputs, outputs])
#         inputs = outputs
#         filters = filters + growth_rate
#     return outputs


# def dense_conv(inputs, filters, layers, bottleneck=False):
#     outputs = []
#     for k in range(layers):
#         if k > 1:
#             inputs = concat()(outputs)
#         if bottleneck:
#             inputs = batch_normalization()(inputs)
#             inputs = relu()(inputs)
#             inputs = conv2d(filters=filters, kernel_size=(1, 1))(inputs)
#         inputs = batch_normalization()(inputs)
#         inputs = relu()(inputs)
#         inputs = conv2d(filters=filters, kernel_size=(3, 3))(inputs)
#         outputs.append(inputs)
#     outputs = concat()(outputs)
#     return outputs


def transition_block(inputs, reduction=0.5):
    inputs = batch_normalization()(inputs)
    inputs = relu()(inputs)
    inputs = conv2d(filters=int(K.int_shape(inputs)[BN_AXIS] * reduction),
                    kernel_size=(1, 1))(inputs)
    inputs = average_pooling2d(pool_size=(2, 2), strides=2)(inputs)

    return inputs


# def transition(inputs, filters):
#     outputs = conv2d(filters, kernel_size=(1, 1))(inputs)
#     outputs = average_pooling2d(pool_size=(2, 2), strides=2)(outputs)
#     return outputs


def build(input_shape):
    """
    Dense Block for Cifar10/Cifar100.

    Num of Dense Block: n = 3
    Num of Transition Block:  n - 1 = 2
    Num of Convolution Layers: L = {40, 100, ...}
    Num of Convolutions in each Dense Block: m = (L - (n - 1 + Num of first Conv + Num of FC)) / (n * 2)
    (Num of first Conv = 1, Num of FC = 1.)
    (each Dense Block has an equal number of layers.)
    m = (L - n -1) / 3
    """

    k = GROWTH_RATE
    n = BLOCKS
    # l = (LAYERS - 4) // 3 // 3
    l = (LAYERS - n - 1) // (2 * n)

    inputs = keras.Input(shape=input_shape)

    outputs = conv2d(16, kernel_size=(3, 3))(inputs)

    for i in range(n):
        outputs = dense_block(outputs, l)
        if i < n - 1:
            outputs = transition_block(outputs)
    #
    # for i in range(n):
    #     # outputs = dense_conv(outputs, filters=k, layers=l)
    #     outputs = dense_conv_block(outputs, k, layers=l)
    #     if i < n - 1:
    #         outputs = transition(outputs, k * l)
    #         # k += GROWTH_RATE

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

    # model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
    #               loss=keras.losses.categorical_crossentropy,
    #               metrics=['acc'])
    #
    # datagen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=4,
    #                                                        height_shift_range=4,
    #                                                        fill_mode='constant',
    #                                                        horizontal_flip=True)
    #
    # datagen.fit(X_train)
    #
    # history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
    #                               steps_per_epoch=TRAIN_SIZE // 10,
    #                               epochs=EPOCHS,
    #                               verbose=2,
    #                               callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, mode='auto')],
    #                               validation_data=(X_val, Y_val))
    #
    # plot(history, metrics=['loss', 'acc'])
    #
    # score = model.evaluate(X_test, Y_test, verbose=0)
    #
    # print("loss: ", score[0])
    # print("acc:  ", score[1])

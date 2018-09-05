# -*- coding: utf-8 -*-

import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K

from utils import plot

CLASSES = 10
ROWS, COLS, CHS = 32, 32, 3

TRAIN_SIZE = 50000
TEST_SIZE = 10000
VALIDATION_SPLIT = TEST_SIZE / TRAIN_SIZE

BATCH_SIZE = 128
EPOCHS = 32


def prepare():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
    X_train = X_train.astype(float) / 255
    X_test = X_test.astype(float) / 255
    Y_train = keras.utils.to_categorical(Y_train, CLASSES)
    Y_test = keras.utils.to_categorical(Y_test, CLASSES)

    return (X_train, Y_train), (X_test, Y_test)


def add():
    return keras.layers.Add()


def batch_normalization():
    return keras.layers.BatchNormalization()


def conv2d(filters, kernel_size, strides=1):
    return keras.layers.Conv2D(filters,
                               kernel_size,
                               strides=strides,
                               padding='same',
                               use_bias=False,
                               kernel_initializer=keras.initializers.he_normal(),
                               kernel_regularizer=keras.regularizers.l2(0.0001))


def dense(units):
    return keras.layers.Dense(units,
                              kernel_regularizer=keras.regularizers.l2(0.0001))


def global_average_pooling2d():
    return keras.layers.GlobalAveragePooling2D()


def relu():
    # return keras.layers.ReLU()
    return keras.layers.Activation(keras.activations.relu)


def softmax():
    # return keras.layers.Softmax()
    return keras.layers.Activation(keras.activations.softmax)


def residual_v1(inputs, filters, kernel_size=(3, 3), down_sampling=False):
    """
    Residual Block v1

    # References
        - [Deep Residual Learning for Image Recognition] (http://arxiv.org/abs/1512.03385)
    """

    if down_sampling:
        outputs = conv2d(filters, kernel_size, strides=2)(inputs)
        inputs = conv2d(filters, (1, 1), strides=2)(inputs)
    else:
        outputs = conv2d(filters, kernel_size)(inputs)
    outputs = batch_normalization()(outputs)
    outputs = relu()(outputs)
    outputs = conv2d(filters, kernel_size)(outputs)
    outputs = add()([inputs, outputs])
    outputs = relu()(outputs)

    return outputs


def residual(inputs, filters, kernel_size=(3, 3), down_sampling=False):
    """
    Residual Block

    # References
        - [Deep Residual Learning for Image Recognition] (http://arxiv.org/abs/1512.03385)
        - [Identity Mappings in Deep Residual Networks] (https://arxiv.org/abs/1603.05027)
    """

    outputs = batch_normalization()(inputs)
    outputs = relu()(outputs)
    if down_sampling:
        outputs = conv2d(filters, kernel_size, strides=2)(outputs)
        inputs = conv2d(filters, (1, 1), strides=2)(inputs)
    else:
        outputs = conv2d(filters, kernel_size)(outputs)
    outputs = batch_normalization()(outputs)
    outputs = relu()(outputs)
    outputs = conv2d(filters, kernel_size)(outputs)
    outputs = add()([inputs, outputs])

    return outputs


def build(input_shape, n=3):
    """

    | output map size | 32x32 | 16 x 16 | 8x8 |
    | # layers        | 1+2n  | 2n      | 2n  |
    | # filters       | 16    | 32      | 64  |
    :param input_shape:
    :param n: Variant for ResNet Cifar-10
            3 => 20 layer
            5 => 32 layer
            7 => 44 layer
            9 => 56 layer
            18 => 110 layer
    :return:
    """
    inputs = keras.Input(shape=input_shape)

    outputs = conv2d(16, kernel_size=(3, 3))(inputs)

    for i in range(n):
        outputs = residual(outputs, 16)

    for i in range(n):
        if i == 0:
            outputs = residual(outputs, 32, (3, 3), down_sampling=True)
        else:
            outputs = residual(outputs, 32, (3, 3))

    for i in range(n):
        if i == 0:
            outputs = residual(outputs, 64, (3, 3), down_sampling=True)
        else:
            outputs = residual(outputs, 64, (3, 3))

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

    model = build(input_shape=(ROWS, COLS, CHS,), n=9)

    model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    # We follow the simple data augmentation in "Deeply-supervised nets" (http://arxiv.org/abs/1409.5185) for training:
    # 4 pixels are padded on each side,
    # and a 32×32 crop is randomly sampled from the padded image or its horizontal flip.
    # For testing, we only evaluate the single view of the original 32×32 image.
    datagen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=4,
                                                           height_shift_range=4,
                                                           fill_mode='constant',
                                                           horizontal_flip=True)

    datagen.fit(X_train)

    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                  steps_per_epoch=TRAIN_SIZE // 10,
                                  epochs=EPOCHS,
                                  verbose=2,
                                  # callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, mode='auto')],
                                  validation_data=(X_val, Y_val))

    plot(history, metrics=['loss', 'acc'])

    score = model.evaluate(X_test, Y_test, verbose=0)

    print("loss: ", score[0])
    print("acc:  ", score[1])

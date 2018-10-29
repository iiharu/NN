# -*- coding: utf-8

from tensorflow import keras

from .__layers__ import (concat, conv2d, dense, flatten, max_pooling2d, sigmoid,
                         softmax)


class LeNet5:

    """
    LeNet-5

    Example:
    model = LeNet5().build()
    """

    def build(self, input_shape=(28, 28, 1), classes=10):
        inputs = keras.Input(shape=input_shape)

        outputs = conv2d(filters=6, kernel_size=(6, 6))(inputs)
        outputs = max_pooling2d(pool_size=(2, 2), strides=(2, 2))(outputs)
        outputs = sigmoid()(outputs)

        outputs = conv2d(filters=16, kernel_size=(6, 6))(inputs)
        outputs = max_pooling2d(pool_size=(2, 2), strides=(2, 2))(outputs)
        outputs = sigmoid()(outputs)

        outputs = flatten()(outputs)

        outputs = dense(120)(outputs)
        outputs = sigmoid()(outputs)

        outputs = dense(64)(outputs)
        outputs = sigmoid()(outputs)

        outputs = dense(classes)(outputs)
        outputs = softmax()(outputs)

        model = keras.Model(inputs, outputs)

        model.summary()

        return model


class LeNet:
    """
    LeNet-300-100

    Example:
    model = LeNet().build()
    """

    def build(self, input_shape=(28, 28, 1), classes=10):
        inputs = keras.Input(shape=input_shape)

        outputs = flatten()(inputs)
        outputs = dense(300)(outputs)
        outputs = sigmoid()(outputs)

        outputs = dense(100)(outputs)
        outputs = sigmoid()(outputs)

        outputs = dense(10)(outputs)
        outputs = softmax()(outputs)

        model = keras.Model(inputs, outputs)

        model.summary()

        return model

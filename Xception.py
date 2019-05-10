# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow import keras

from layers import (add, batch_normalization, conv2d, dense,
                    global_average_pooling2d, max_pooling2d, relu,
                    separable_conv2d, softmax)


def block(inputs, layers):
    for layer in layers:
        inputs = layer(inputs)
    return inputs


def conv_block(inputs, filters,
               kernel_size=(3, 3), strides=1):
    inputs = conv2d(filters=filters, kernel_size=kernel_size,
                    strides=strides)(inputs)
    inputs = batch_normalization()(inputs)
    return inputs


def conv_relu_block(inputs, filters,
                    kernel_size=(3, 3), strides=1):
    inputs = conv_block(inputs, filters=filters,
                        kernel_size=kernel_size, strides=strides)
    inputs = relu()(inputs)
    return inputs


def separable_conv_block(inputs, filters,
                         kernel_size=(3, 3), strides=1):
    inputs = separable_conv2d(filters=filters, kernel_size=kernel_size)(inputs)
    inputs = batch_normalization()(inputs)
    return inputs


def separable_conv_relu_block(inputs, filters,
                              kernel_size=(3, 3), strides=1):
    inputs = separable_conv_block(inputs, filters=filters,
                                  kernel_size=kernel_size, strides=strides)
    inputs = relu()(inputs)
    return inputs


def relu_separable_conv_block(inputs, filters,
                              kernel_size=(3, 3), strides=1):
    inputs = relu()(inputs)
    inputs = separable_conv_block(inputs, filters=filters,
                                  kernel_size=kernel_size, strides=strides)
    inputs = relu()(inputs)
    return inputs


class Xception:
    """
    Xception

    - Referecences:
      - [Xception: Deep Learning with Depthwise Separable Convolutions] (https://arxiv.org/abs/1610.02357)
    """

    def build(self, input_shape, classes):

        inputs = keras.Input(shape=input_shape)

        # Entry flow
        filters = 32
        outputs = conv_relu_block(inputs, filters=filters, strides=(2, 2))

        filters = 64
        outputs = conv_relu_block(inputs, filters=filters)

        residual = outputs

        filters = 128
        outputs = separable_conv_block(outputs, filters=filters)
        outputs = relu_separable_conv_block(outputs, filters=filters)
        outputs = max_pooling2d(pool_size=(3, 3), strides=(2, 2))(outputs)
        residual = conv_block(residual, filters=filters,
                              kernel_size=(1, 1), strides=(2, 2))
        outputs = add()([outputs, residual])

        filters = 256
        outputs = relu_separable_conv_block(outputs, filters=filters)
        outputs = relu_separable_conv_block(outputs, filters=filters)
        outputs = max_pooling2d(pool_size=(3, 3), strides=(2, 2))(outputs)
        residual = conv_block(residual, filters=filters,
                              kernel_size=(1, 1), strides=(2, 2))
        outputs = add()([outputs, residual])

        filters = 728
        outputs = relu_separable_conv_block(outputs, filters=filters)
        outputs = relu_separable_conv_block(outputs, filters=filters)
        outputs = max_pooling2d(pool_size=(3, 3), strides=(2, 2))(outputs)
        residual = conv_block(residual, filters=filters,
                              kernel_size=(1, 1), strides=(2, 2))
        outputs = add()([outputs, residual])

        # Middle flow
        filters = 728
        for _ in range(8):
            residual = outputs

            outputs = relu_separable_conv_block(outputs, filters=filters)
            outputs = relu_separable_conv_block(outputs, filters=filters)
            outputs = relu_separable_conv_block(outputs, filters=filters)
            outputs = add()([outputs, residual])

        # Exit flow
        residual = outputs
        outputs = relu_separable_conv_block(outputs, filters=filters)

        filters = 1024
        outputs = relu_separable_conv_block(outputs, filters=filters)
        outputs = add()([outputs, residual])

        filters = 1536
        outputs = separable_conv_relu_block(outputs, filters=filters)

        filters = 2048
        outputs = separable_conv_relu_block(outputs, filters=filters)

        outputs = global_average_pooling2d()(outputs)

        outputs = dense(filters)(outputs)
        outputs = relu()(outputs)

        outputs = dense(1000)(outputs)
        outputs = softmax()(outputs)

        model = keras.Model(inputs, outputs)

        model.summary()

        return model

# -*- coding: utf-8 -*-

from tensorflow import keras

from .__layers__ import conv2d, max_pooling2d, concat, batch_normalization, relu, dense, softmax


def CONV(inputs, filters, kernel_size, strides=1):
    outputs = batch_normalization()(inputs)
    outputs = relu()(outputs)
    outputs = conv2d(filters=filters, kernel_size=kernel_size,
                     strides=strides)(inputs)
    return outputs


def conv_block(inputs, filters, kernel_size, strides=1, activation=None, use_bias=True,
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
               kernel_constraint=None, bias_constraint=None):
    inputs = batch_normalization()(inputs)
    inputs = relu()(inputs)
    inputs = conv2d(filters=filters, kernel_size=kernel_size, strides=strides, activation=activation, use_bias=True,
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(inputs)
    return inputs


def fc_softmax(inputs, classes, use_bias=True,
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None, bias_constraint=None):
    inputs = dense(classes, use_bias=use_bias,
                   kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                   kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                   activity_regularizer=activity_regularizer,
                   kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(inputs)
    inputs = softmax()(inputs)
    return inputs


class GoogLeNet:
    """
    GoogLeNet
    """

    def __init__(self):
        self.block = 2

    def inception(self, inputs, filters, reduction=False):
        outputs = self.inception_v1(inputs, filters, reduction=reduction)
        return outputs

    def inception_v1(self, inputs, filters, reduction=False):
        outputs1 = inputs
        outputs2 = inputs
        outputs3 = inputs
        outputs4 = inputs

        strides = 2 if reduction else 1
        outputs1 = conv_block(inputs=outputs1, filters=filters, kernel_size=(1, 1),
                              strides=strides)

        if reduction:
            outputs2 = conv_block(inputs=outputs2,
                                  filters=filters, kernel_size=(1, 1))
        outputs2 = conv_block(inputs=outputs2,
                              filters=filters, kernel_size=(3, 3))

        if reduction:
            outputs3 = conv_block(inputs=outputs3,
                                  filters=filters, kernel_size=(1, 1), strides=2)
        outputs3 = conv_block(inputs=outputs3,
                              filters=filters, kernel_size=(3, 3))

        outputs4 = max_pooling2d(pool_size=(3, 3), strides=2)(outputs4)
        if reduction:
            outputs4 = conv_block(inputs=outputs4,
                                  filters=filters, kernel_size=(3, 3))

        outputs = concat()([outputs1, outputs2, outputs3, outputs4])

        return outputs

# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow import keras
from layers import flatten, max_pooling2d, relu, softmax


def conv2d(filters, kernel_size, strides=1,
           activation=None, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None,
           **kwargs):
    return keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                  activation=activation, use_bias=use_bias,
                  kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                  kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                  kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                  **kwargs)


def dense(units, activation=None, use_bias=True,
          kernel_initializer='glorot_uniform', bias_initializer='zeros',
          kernel_regularizer=None, bias_regularizer=None,
          activity_regularizer=None,
          kernel_constraint=None, bias_constraint=None,
          **kwargs):
    return keras.layers.Dense(units, activation=activation, use_bias=use_bias,
                 kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                 kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                 activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                 **kwargs)


class VGG:

    """
    VGG

    Example:
    # VGG16
    model = VGG(layers=16, blocks=[2, 2, 3, 3, 3], filters=[64, 128, 256, 512, 512]).build()
    # VGG19
    model = VGG(layers=19, blocks=[2, 2, 4, 4, 4], filters=[64, 128, 256, 512, 512]).build()
    """

    def __init__(self, layers=16, blocks=[2, 2, 3, 3, 3], filters=[64, 128, 256, 512, 512]):
        self.layers = layers
        self.blocks = blocks
        self.filters = filters

    def conv_block(self, inputs, filters, blocks):
        for _ in range(blocks):
            inputs = conv2d(filters=filters, kernel_size=(3, 3))(inputs)
            inputs = relu()(inputs)
        return inputs

    def build(self, input_shape=(224, 224, 3), classes=1000):
        inputs = keras.Input(shape=input_shape)

        outputs = inputs
        for i, b in enumerate(self.blocks):
            outputs = self.conv_block(outputs, self.filters[i], b)
            outputs = max_pooling2d(pool_size=(2, 2))(outputs)

        outputs = dense(4096)(outputs)
        outputs = dense(4096)(outputs)

        outputs = dense(1000)(outputs)
        outputs = softmax()(outputs)

        model = keras.Model(inputs, outputs)

        model.summary()

        return model

if __name__ == '__main__':
    model = VGG().build()

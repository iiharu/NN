# -*- coding: utf-8

from tensorflow import keras
from tensorflow.keras import layers


def concat(axis=-1, **kwargs):
    return layers.Concatenate(axis=axis, **kwargs)


def conv2d(filters, kernel_size, strides=1, padding='same',
           activation=None, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None,
           **kwargs):
    return layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
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
    return layers.Dense(units, activation=activation, use_bias=use_bias,
                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
                        kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                        **kwargs)


def flatten(**kwargs):
    return layers.Flatten(**kwargs)


def max_pooling2d(pool_size=(2, 2), strides=None, padding='same', **kwargs):
    return layers.MaxPooling2D(pool_size=pool_size,
                               strides=strides,
                               padding=padding,
                               **kwargs)


def sigmoid(**kwargs):
    return layers.Activation(activation='sigmoid', **kwargs)


def softmax(**kwargs):
    return layers.Activation(activation='softmax', **kwargs)


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


if __name__ == '__main__':
    model = LeNet().build()
    model = LeNet5().build()

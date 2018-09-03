# -*- coding: utf-8 -*-

from tensorflow import keras


"""
Layer function wrappers for Keras Layer.

Do not import all functions in this modules,
because functions having many parameter is not useful with multiple use.
"""

def add():
    return keras.layers.Add()


def average_pooling2d(pool_size=(2, 2), strides=None):
    return keras.layers.AveragePooling2D(pool_size=pool_size,
                                         strides=strides,
                                         padding='same')


def batch_normalization():
    return keras.layers.BatchNormalization()


def concat(axis=-1):
    return keras.layers.Concatenate(axis=axis)


def conv2d(filters, kernel_size, strides=1,
           activation=None, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None):
    return keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                               activation=activation, use_bias=use_bias,
                               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                               kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                               kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)


def dense(units, activation=None, use_bias=True,
          kernel_initializer='glorot_uniform', bias_initializer='zeros',
          kernel_regularizer=None, bias_regularizer=None,
          activity_regularizer=None,
          kernel_constraint=None, bias_constraint=None):
    return keras.layers.Dense(units, activation=activation, use_bias=use_bias,
                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                              kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                              activity_regularizer=activity_regularizer,
                              kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)


def dropout(rate, noise_shape=None):
    return keras.layers.Dropout(rate, noise_shape=noise_shape)


def flatten():
    return keras.layers.Flatten()


def global_average_pooling2d():
    return keras.layers.GlobalAveragePooling2D()


def max_pooling2d(pool_size=(2, 2), strides=None):
    return keras.layers.MaxPooling2D(pool_size=pool_size,
                                     strides=strides,
                                     padding='same')


def relu(max_value=None):
    # return keras.layers.ReLU(max_value=max_value)
    return keras.layers.Activation(keras.activations.relu)


def softmax(axis=-1):
    # return keras.layers.Softmax(axis=axis)
    return keras.layers.Activation(keras.activations.softmax)

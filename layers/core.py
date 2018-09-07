# -*- coding: utf-8 -*-

from tensorflow import keras


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

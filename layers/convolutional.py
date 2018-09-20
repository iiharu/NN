# -*- coding: utf-8 -*-

from tensorflow import keras


def conv1d(filters, kernel_size, strides=1,
           activation=None, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None,
           activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None,
           **kwargs):
    return keras.layers.Conv1D(filters, kernel_size, strides=1, padding='same',
                               activation=activation, use_bias=use_bias,
                               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                               kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                               activity_regularizer=activity_regularizer,
                               kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                               **kwargs)


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


def cropping1d(cropping=(1, 1), **kwargs):
    return keras.layers.Cropping1D(cropping, **kwargs)


def cropping2d(cropping=((0, 0), (0, 0)), **kwargs):
    return keras.layers.Cropping2D(cropping, **kwargs)


def transposed_conv2d(filters, kernel_size, strides=1,
                      activation=None, use_bias=True,
                      kernel_initializer='glorot_uniform', bias_initializer='zeros',
                      kernel_regularizer=None, bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None, bias_constraint=None,
                      **kwargs):
    return keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same',
                                        activation=activation, use_bias=use_bias,
                                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                        kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                        activity_regularizer=activity_regularizer,
                                        kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                        **kwargs)


def up_sampling1d(size=2, **kwargs):
    return keras.layers.UpSampling1D(size, **kwargs)


def up_sampling2d(size=(2, 2), **kwargs):
    return keras.layers.UpSampling2D(size, **kwargs)


def zero_padding1d(padding=1, **kwargs):
    return keras.layers.ZeroPadding1D(padding=padding, **kwargs)


def zero_padding2d(padding=(1, 1), **kwargs):
    return keras.layers.ZeroPadding2D(padding=padding, **kwargs)
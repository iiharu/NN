# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Cropping1D
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import ZeroPadding1D
from tensorflow.keras.layers import ZeroPadding2D


def conv1d(filters, kernel_size, strides=1,
           activation=None, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None,
           activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None,
           **kwargs):
    return Conv1D(filters, kernel_size, strides=1, padding='same',
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
    return Conv2D(filters, kernel_size, strides=strides, padding='same',
                  activation=activation, use_bias=use_bias,
                  kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                  kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                  kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                  **kwargs)


def cropping1d(cropping=(1, 1), **kwargs):
    return Cropping1D(cropping, **kwargs)


def cropping2d(cropping=((0, 0), (0, 0)), **kwargs):
    return Cropping2D(cropping, **kwargs)


def transposed_conv2d(filters, kernel_size, strides=1,
                      activation=None, use_bias=True,
                      kernel_initializer='glorot_uniform', bias_initializer='zeros',
                      kernel_regularizer=None, bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None, bias_constraint=None,
                      **kwargs):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding='same',
                           activation=activation, use_bias=use_bias,
                           kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                           activity_regularizer=activity_regularizer,
                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                           **kwargs)


def up_sampling1d(size=2, **kwargs):
    return UpSampling1D(size, **kwargs)


def up_sampling2d(size=(2, 2), **kwargs):
    return UpSampling2D(size, **kwargs)


def zero_padding1d(padding=1, **kwargs):
    return ZeroPadding1D(padding=padding, **kwargs)


def zero_padding2d(padding=(1, 1), **kwargs):
    return ZeroPadding2D(padding=padding, **kwargs)

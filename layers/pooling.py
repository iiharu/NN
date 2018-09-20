# -*- coding: utf-8 -*-

from tensorflow import keras


def average_pooling1d(pool_size=2, strides=None, **kwargs):
    return keras.layers.AveragePooling1D(pool_size=pool_size,
                                         strides=strides,
                                         padding='same',
                                         **kwargs)


def average_pooling2d(pool_size=(2, 2), strides=None, **kwargs):
    return keras.layers.AveragePooling2D(pool_size=pool_size,
                                         strides=strides,
                                         padding='same',
                                         **kwargs)


def global_average_pooling1d(**kwargs):
    return keras.layers.GlobalAveragePooling1D(**kwargs)


def global_average_pooling2d(**kwargs):
    return keras.layers.GlobalAveragePooling2D(**kwargs)


def max_pooling1d(pool_size=2, strides=None, **kwargs):
    return keras.layers.MaxPooling1D(pool_size=pool_size,
                                     strides=strides,
                                     padding='same',
                                     **kwargs)


def max_pooling2d(pool_size=(2, 2), strides=None, **kwargs):
    return keras.layers.MaxPooling2D(pool_size=pool_size,
                                     strides=strides,
                                     padding='same',
                                     **kwargs)

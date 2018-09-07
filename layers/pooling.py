# -*- coding: utf-8 -*-

from tensorflow import keras

def average_pooling2d(pool_size=2, strides=None):
    return keras.layers.AveragePooling1D(pool_size=pool_size,
                                         strides=strides,
                                         padding='same')


def average_pooling2d(pool_size=(2, 2), strides=None):
    return keras.layers.AveragePooling2D(pool_size=pool_size,
                                         strides=strides,
                                         padding='same')


def global_average_pooling1d():
    return keras.layers.GlobalAveragePooling1D()


def global_average_pooling2d():
    return keras.layers.GlobalAveragePooling2D()


def max_pooling1d(pool_size=2, strides=None):
    return keras.layers.MaxPooling1D(pool_size=pool_size,
                                     strides=strides,
                                     padding='same')


def max_pooling2d(pool_size=(2, 2), strides=None):
    return keras.layers.MaxPooling2D(pool_size=pool_size,
                                     strides=strides,
                                     padding='same')
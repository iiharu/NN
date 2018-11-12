# -*- coding: utf-8 -*-

from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D


def average_pooling2d(pool_size=(2, 2), strides=None, padding='same', **kwargs):
    return AveragePooling2D(pool_size=pool_size,
                            strides=strides,
                            padding=padding,
                            **kwargs)


def global_average_pooling2d(**kwargs):
    return GlobalAveragePooling2D(**kwargs)


def max_pooling2d(pool_size=(2, 2), strides=None, padding='same', **kwargs):
    return MaxPooling2D(pool_size=pool_size,
                        strides=strides,
                        padding=padding,
                        **kwargs)

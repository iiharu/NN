# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras


def average_pooling2d(pool_size=(2, 2), strides=None, padding='same', **kwargs):
    return keras.layers.AveragePooling2D(pool_size=pool_size,
                                         strides=strides,
                                         padding=padding,
                                         **kwargs)


def global_average_pooling2d(**kwargs):
    return keras.layers.GlobalAveragePooling2D(**kwargs)


def max_pooling2d(pool_size=(2, 2), strides=None, padding='same', **kwargs):
    return keras.layers.MaxPooling2D(pool_size=pool_size,
                                     strides=strides,
                                     padding=padding,
                                     **kwargs)

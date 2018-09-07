# -*- coding: utf-8 -*-

from tensorflow import keras


def batch_normalization(axis=-1):
    return keras.layers.BatchNormalization(axis=axis)

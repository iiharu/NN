# -*- coding: utf-8 -*-

from tensorflow import keras


def add(**kwargs):
    return keras.layers.Add(**kwargs)


def concat(axis=-1, **kwargs):
    return keras.layers.Concatenate(axis=axis, **kwargs)

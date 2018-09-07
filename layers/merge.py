# -*- coding: utf-8 -*-

from tensorflow import keras


def add():
    return keras.layers.Add()


def concat(axis=-1):
    return keras.layers.Concatenate(axis=axis)

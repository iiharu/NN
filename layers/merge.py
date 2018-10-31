# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Concatenate


def add(**kwargs):
    return Add(**kwargs)


def concat(axis=-1, **kwargs):
    return Concatenate(axis=axis, **kwargs)

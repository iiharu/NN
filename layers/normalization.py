# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras


def batch_normalization(axis=-1, **kwargs):
    return keras.layers.BatchNormalization(axis=axis, **kwargs)

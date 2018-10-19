# -*- coding: utf-8 -*-

from tensorflow.keras.layers import BatchNormalization


def batch_normalization(axis=-1, **kwargs):
	return BatchNormalization(axis=axis, **kwargs)

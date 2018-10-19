# -*- coding: utf-8 -*-

from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import MaxPooling2D


def average_pooling1d(pool_size=2, strides=None, **kwargs):
	return AveragePooling1D(pool_size=pool_size,
							strides=strides,
							padding='same',
							**kwargs)


def average_pooling2d(pool_size=(2, 2), strides=None, **kwargs):
	return AveragePooling2D(pool_size=pool_size,
							strides=strides,
							padding='same',
							**kwargs)


def global_average_pooling1d(**kwargs):
	return GlobalAveragePooling1D(**kwargs)


def global_average_pooling2d(**kwargs):
	return GlobalAveragePooling2D(**kwargs)


def max_pooling1d(pool_size=2, strides=None, **kwargs):
	return MaxPooling1D(pool_size=pool_size,
						strides=strides,
						padding='same',
						**kwargs)


def max_pooling2d(pool_size=(2, 2), strides=None, **kwargs):
	return MaxPooling2D(pool_size=pool_size,
						strides=strides,
						padding='same',
						**kwargs)

# -*- coding: utf-8 -*-

from tensorflow import keras

from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D


def concat(axis=-1, **kwargs):
	return Concatenate(axis=axis, **kwargs)


def conv2d(filters, kernel_size, strides=1,
		   activation=None, use_bias=True,
		   kernel_initializer='glorot_uniform', bias_initializer='zeros',
		   kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
		   kernel_constraint=None, bias_constraint=None,
		   **kwargs):
	return Conv2D(filters, kernel_size, strides=strides, padding='same',
				  activation=activation, use_bias=use_bias,
				  kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
				  kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
				  kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
				  **kwargs)


def max_pooling2d(pool_size=(2, 2), strides=None, **kwargs):
	return MaxPooling2D(pool_size=pool_size,
						strides=strides,
						padding='same',
						**kwargs)

def inception_v1(inputs, filters, reduction=False):
	outputs1 = inputs
	outputs2 = inputs
	outputs3 = inputs
	outputs4 = inputs

	if reduction:
		outputs1 = conv2d(filters=filters, kernel_size=(1, 1), strides=2)(outputs1)
	else:
		outputs1 = conv2d(filters=filters, kernel_size=(1, 1))(outputs1)

	if reduction:
		outputs2 = conv2d(filters=filters, kernel_size=(1, 1), strides=2)(outputs2)
	outputs2 = conv2d(filters=filters, kernel_size=(3, 3))(outputs2)

	if reduction:
		outputs3 = conv2d(filters=filters, kernel_size=(1, 1), strides=2)(outputs3)
	outputs3 = conv2d(filters=filters, kernel_size=(5, 5))(outputs3)

	outputs4 = max_pooling2d(pool_size=(3, 3), strides=1)(outputs4)
	if reduction:
		outputs4 = conv2d(filters=filters, kernel_size=(1, 1), strides=2)(outputs4)

	outputs = concat()([outputs1, outputs2, outputs3, outputs4])

	return outputs


def 


if __name__ == '__main__':

	inputs = keras.Input(shape=(32, 32, 3))
	outputs = inception_v1(inputs, filters=8, reduction=True)
	outputs = inception_v1(outputs, filters=8)

	model = keras.Model(inputs, outputs)

	model.summary()

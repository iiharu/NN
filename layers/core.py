# -*- coding: utf-8 -*-

from tensorflow import keras


def add():
    return keras.layers.Add()


def average_pooling2d(pool_size=2, strides=None):
    return keras.layers.AveragePooling1D(pool_size=pool_size,
                                         strides=strides,
                                         padding='same')


def average_pooling2d(pool_size=(2, 2), strides=None):
    return keras.layers.AveragePooling2D(pool_size=pool_size,
                                         strides=strides,
                                         padding='same')


def batch_normalization():
    return keras.layers.BatchNormalization()


def concat(axis=-1):
    return keras.layers.Concatenate(axis=axis)


def conv1d(filters, kernel_size, strides=1,
           activation=None, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None,
           activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None):
    return keras.layers.Conv1D(filters, kernel_size, strides=1, padding='same',
                               activation=activation, use_bias=use_bias,
                               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                               kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                               activity_regularizer=activity_regularizer,
                               kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)


def conv2d(filters, kernel_size, strides=1,
           activation=None, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None):
    return keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                               activation=activation, use_bias=use_bias,
                               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                               kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                               kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)


def cropping1d(cropping=(1, 1)):
    return keras.layers.Cropping1D(cropping)


def cropping2d(cropping=((0, 0), (0, 0))):
    return keras.layers.Cropping2D(cropping)


def dense(units, activation=None, use_bias=True,
          kernel_initializer='glorot_uniform', bias_initializer='zeros',
          kernel_regularizer=None, bias_regularizer=None,
          activity_regularizer=None,
          kernel_constraint=None, bias_constraint=None):
    return keras.layers.Dense(units, activation=activation, use_bias=use_bias,
                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                              kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                              activity_regularizer=activity_regularizer,
                              kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)


def dropout(rate, noise_shape=None):
    return keras.layers.Dropout(rate, noise_shape=noise_shape)


def flatten():
    return keras.layers.Flatten()


def global_average_pooling1d():
    return keras.layers.GlobalAveragePooling1D()


def global_average_pooling2d():
    return keras.layers.GlobalAveragePooling2D()


def max_pooling1d(pool_size=2, strides=None):
    return keras.layers.MaxPooling1D(pool_size=pool_size,
                                     strides=strides,
                                     padding='same')


def max_pooling2d(pool_size=(2, 2), strides=None):
    return keras.layers.MaxPooling2D(pool_size=pool_size,
                                     strides=strides,
                                     padding='same')


def relu(max_value=None):
    # return keras.layers.ReLU(max_value=max_value)
    return keras.layers.Activation(keras.activations.relu)


def softmax(axis=-1):
    # return keras.layers.Softmax(axis=axis)
    return keras.layers.Activation(keras.activations.softmax)


def up_sampling1d(size=2):
    return keras.layers.UpSampling1D(size)


def up_sampling2d(size=(2, 2)):
    return keras.layers.UpSampling2D(size)


def transposed_conv2d(filters, kernel_size, strides=1,
                      activation=None, use_bias=True,
                      kernel_initializer='glorot_uniform', bias_initializer='zeros',
                      kernel_regularizer=None, bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None, bias_constraint=None):
    return keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same',
                                        activation=activation, use_bias=use_bias,
                                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                        kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                        activity_regularizer=activity_regularizer,
                                        kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)


def zero_padding1d(padding=1):
    return keras.layers.ZeroPadding1D(padding=padding)


def zero_padding2d(padding=(1, 1)):
    return keras.layers.ZeroPadding2D(padding=padding)
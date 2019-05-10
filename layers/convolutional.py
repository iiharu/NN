# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras


def conv2d(filters, kernel_size, strides=1, padding='same',
           activation=None, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None,
           **kwargs):
    return keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                               activation=activation, use_bias=use_bias,
                               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                               kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                               kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                               **kwargs)


def cropping2d(cropping=((0, 0), (0, 0)), **kwargs):
    return keras.layers.Cropping2D(cropping, **kwargs)


def transposed_conv2d(filters, kernel_size, strides=1, padding='same',
                      activation=None, use_bias=True,
                      kernel_initializer='glorot_uniform', bias_initializer='zeros',
                      kernel_regularizer=None, bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None, bias_constraint=None,
                      **kwargs):
    return keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding,
                                        activation=activation, use_bias=use_bias,
                                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                        kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                        activity_regularizer=activity_regularizer,
                                        kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                        **kwargs)


def up_sampling2d(size=(2, 2), **kwargs):
    return keras.layers.UpSampling2D(size, **kwargs)


def separable_conv2d(filters, kernel_size, strides=(1, 1), padding='same',
                     depth_multiplier=1,
                     activation=None, use_bias=True,
                     depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros',
                     depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                     depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None):
    return keras.layers.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding=padding,
                                        depth_multiplier=depth_multiplier,
                                        activation=activation, use_bias=use_bias,
                                        depthwise_initializer=depthwise_initializer, pointwise_initializer=pointwise_initializer, bias_initializer=bias_initializer,
                                        depthwise_regularizer=depthwise_regularizer, pointwise_regularizer=pointwise_regularizer,
                                        bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                                        depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)


def zero_padding2d(padding=(1, 1), **kwargs):
    return keras.layers.ZeroPadding2D(padding=padding, **kwargs)

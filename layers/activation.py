# -*- coding: utf-8 -*-

from tensorflow import keras


def linear():
    return keras.layers.Activation(keras.activations.linear)


def relu(max_value=None, **kwargs):
    # return keras.layers.ReLU(max_value=max_value, **kwargs)
    return keras.layers.Activation(keras.activations.relu)


def sigmoid():
    return keras.layers.Activation(keras.activations.sigmoid)


def softmax(axis=-1, **kwargs):
    # return keras.layers.Softmax(axis=axis, **kwargs)
    return keras.layers.Activation(keras.activations.softmax)


def tanh():
    return keras.layers.Activation(keras.activations.tanh)

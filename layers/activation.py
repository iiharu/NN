# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Activation
from tensorflow.keras import activations


def linear():
    return Activation(activations.linear)


def relu(max_value=None, **kwargs):
    # return keras.layers.ReLU(max_value=max_value, **kwargs)
    return Activation(activations.relu)


def sigmoid():
    return Activation(activations.sigmoid)


def softmax(axis=-1, **kwargs):
    # return keras.layers.Softmax(axis=axis, **kwargs)
    return Activation(activations.softmax)


def tanh():
    return Activation(activations.tanh)

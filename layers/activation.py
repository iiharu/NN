# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Activation
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer


class ReLU(Layer):
    def __init__(self, **kwargs):
        super(ReLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return activations.relu(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


class Softmax(Layer):
    def __init__(self, **kwargs):
        super(Softmax, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return activations.softmax(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


def activation(activation, **kwargs):
    return Activation(activation=activation, **kwargs)


def elu(**kwargs):
    return Activation(activation=activations.elu, **kwargs)


def linear(**kwargs):
    return Activation(activation=activations.linear, **kwargs)


def relu(max_value=None, **kwargs):
    # return keras.layers.ReLU(max_value=max_value, **kwargs)
    return Activation(activation=activations.relu, **kwargs)


def selu(**kwargs):
    return Activation(activation=activations.selu, **kwargs)


def sigmoid(**kwargs):
    return Activation(activation=activations.sigmoid, **kwargs)


def softmax(axis=-1, **kwargs):
    # return keras.layers.Softmax(axis=axis, **kwargs)
    return Activation(activation=activations.softmax, **kwargs)


def tanh(**kwargs):
    return Activation(activation=activations.tanh, **kwargs)

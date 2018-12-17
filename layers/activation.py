# -*- coding: utf-8 -*-

from functools import partial
from tensorflow.keras.layers import Activation
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer


class ReLU(Layer):
    def __init__(self, **kwargs):
        super(ReLU, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(ReLU, self).build(input_shape)

    def call(self, inputs):
        return activations.relu(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


class Softmax(Layer):
    def __init__(self, **kwargs):
        super(Softmax, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Softmax, self).build(input_shape)

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
    return ReLU(**kwargs)


def selu(**kwargs):
    return Activation(activation=activations.selu, **kwargs)


def sigmoid(**kwargs):
    return Activation(activation=activations.sigmoid, **kwargs)


def softmax(axis=-1, **kwargs):
    return Softmax(**kwargs)


def tanh(**kwargs):
    return Activation(activation=activations.tanh, **kwargs)

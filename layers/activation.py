# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras


def linear(**kwargs):
    return keras.layers.Activation(activation=keras.activations.linear, **kwargs)


def relu(**kwargs):
    return keras.layers.Activation(activation=keras.activations.relu, **kwargs)


def sigmoid(**kwargs):
    return keras.layers.Activation(activation=keras.activations.sigmoid, **kwargs)


def softmax(**kwargs):
    return keras.layers.Activation(activation=keras.activations.softmax, **kwargs)


def tanh(**kwargs):
    return keras.layers.Activation(activaion=keras.activations.tanh, **kwargs)

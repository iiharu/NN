# -*- coding: utf-8 -*-

from tensorflow import keras

from .layers import

def build(input_shape=(224, 224, 3), classes=1000):
    inputs = keras.Input(shape=input_shape)
    outputs = keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='same')(inputs)
# -*- coding: utf-8 -*-

from tensorflow import keras

from .layers import

def build(input_shape=(224, 224, 3), classes=1000):
    inputs = keras.Input(shape=input_shape)
    outputs = keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='same')(inputs)
    # Local Response Normalization
    outputs = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(outputs)
    outputs = keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same')(outputs)
    # Local Response Normalization

    

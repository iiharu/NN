#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

class Standardization(Layer):
    def __init__(self,**kwargs):
        super(Standardization,self).__init__(**kwargs)

    def build(self,input_shape):
        super(Standardization,self).build(input_shape)

    def call(self,x):
        mean = K.mean(x, axis=0)
        var = K.mean(K.square(x - mean),axis=0)
        y = (x - mean) / (K.sqrt(var) + K.epsilon())
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

class BatchNorm(Layer):
    def __init__(self,**kwargs):
        super(BatchNorm, self).__init__(**kwargs)


    def build(self,input_shape):
        self.mean = self.add_weight(name='mean',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=False)
        self.var = self.add_weight(name='var',
                                   shape=input_shape[1:],
                                   initializer='ones',
                                   trainable=False)
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[1:],
                                     initializer='ones',
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)
        super(BatchNorm,self).build(input_shape)

    def call(self, x):
        self.mean, self.var = K.mean(x, axis=0), K.var(x, axis=0)
        y = (x - self.mean) / (K.sqrt(self.var) + K.epsilon())
        y = self.gamma * y + self.beta
        return y

    def compute_output_shape(self,input_shape):
        return input_shape


BATCH_SIZE=32
EPOCHS=10
(X_train, Y_train),(X_test,Y_test) = keras.datasets.mnist.load_data()
X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255
Y_train = keras.utils.to_categorical(Y_train, 10)
Y_test = keras.utils.to_categorical(Y_test, 10)

model = keras.Sequential()
model.add(Flatten(input_shape=(28,28)))
# model.add(BatchNormalization())
model.add(BatchNorm())
model.add(Dense(100, activation='relu'))
# model.add(BatchNormalization())
model.add(BatchNorm())
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='SGD', loss='categorical_crossentropy')

model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE,epochs=EPOCHS)

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
    def build(self,input_shape,**kwargs):
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[1:],
                                     initializer='uniform',
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[1:],
                                    initializer='uniform',
                                    trainable=True)
        super(BatchNorm,self).__init__(**kwargs)
    def call(self, x):
        mean = K.mean(x, axis=0)
        var = K.mean(K.square(x - mean), axis=0)
        y = (x - mean) / (K.sqrt(var) + K.epsilon())
        y = self.gamma * x + self.beta
        return y
    def compute_output_shape(self,input_shape):
        return input_shape


BATCH_SIZE=32
EPOCHS=10
(X_train, Y_train),(X_test,Y_test) = keras.datasets.mnist.load_data()
Y_train = keras.utils.to_categorical(Y_train, 10)
Y_test = keras.utils.to_categorical(Y_test, 10)

model = keras.Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(BatchNormalization())
# model.add(BatchNorm())
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
# model.add(BatchNorm())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='SGD', loss='categorical_crossentropy')

model.fit(X_train, Y_train,batch_size=BATCH_SIZE,epochs=EPOCHS)

# x = K.random_uniform_variable(shape=(10, 5), low=0, high=1)
# print(K.dtype(x))
# print(x.shape)
# mean = K.mean(x,axis=0)
# print(mean.shape)
# var = K.mean(K.square(x - mean), axis=0)
# print(var.shape)
# standardized = (x - mean) / (K.sqrt(var) + K.epsilon())
# print(y.shape)
# print(K.eval(standardized))
# print(x.shape[1:])
# print(x.shape[1:-1])
# gamma = K.variable(np.ones(x.shape[-1]), dtype=K.dtype(x), name='gamma')
# beta = K.variable(np.zeros(x.shape[-1]), dtype=K.dtype(x), name='beta')
# batch_normalized = K.batch_normalization(x, mean, var, beta, gamma)
# y = K.normalize_batch_in_training(x, gamma, beta, reduction_axes=range(x.shape[-1]))
# print(K.eval(batch_normalized))
# print(K.eval(batch_normalized - standardized))

# x = np.random.rand(10, 5)
# mean = np.mean(x, axis=0)
# var = np.mean(np.square(x - mean),axis=0)
# y = (x - mean) / np.sqrt(var)
# print(x)
# print(mean)
# print(var)
# print(y)

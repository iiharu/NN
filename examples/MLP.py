#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

class Standardization(keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Standardization,self).__init__(**kwargs)

    def build(self,input_shape):
        self.mean = self.add_weight(name='mean',
                                    shape=input_shape[1:],
                                    trainable=False)
        self.var = self.add_weight(name='var',
                                   shape=input_shape[1:],
                                   trainable=False)
        super(Standardization,self).build(input_shape)

    def call(self,x):
        # print(x)
        # print(x.shape)
        # mean = keras.backend.mean(x, axis=0)
        self.mean = K.mean(x, axis=0)
        # print(mean.shape)
        # var = keras.backend.mean(keras.backend.square(x - mean),0)
        self.var = K.mean(K.square(x - self.mean), 0)
        # print(var.shape)
        y = (x - self.mean) / (K.sqrt(self.var) + K.epsilon())
        # y = (x - mean) / (keras.backend.sqrt(var) + keras.backend.epsilon())
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

# class BatchNorm(keras.Layer):
#     def __init__(self,**kwargs):
#         super(BatchNorm,self).__init__(**kwargs)
#     def build(self,input_shape):
#         self.ganma = self.add_weight(name='ganma',
#                                      shape=(input_shape[-1]),
#                                      initilizer='uniform',
#                                      trainable=True)
#         self.beta = self.add_weight(name='beta',shape=(input_shape[-1]),
#                                                        initilizer='uniform',
#                                                        trainable=True)
#         self.mu = self.add_weight(name='mu',shape=(input_shape[-1]))


batch_size = 10
steps = 1
X = np.random.rand(100,5,5)
model = keras.Sequential()
model.add(Standardization())
Y = model.predict(X, batch_size=batch_size)
print(Y)

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

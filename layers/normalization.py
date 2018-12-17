# -*- coding: utf-8 -*-

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import BatchNormalization


def batch_normalization(axis=-1, **kwargs):
    return BatchNormalization(axis=axis, **kwargs)


# class Standardization():
#     def __init__(self, **kwargs):
#         super(Standardization, self).__init__(**kwargs)
#     def build(self, input_shape):
#         self.mean = self.add_weight(name='mean',
#                                     shape=input_shape[1:],
#                                     initializer='zeros',
#                                     trainable=False)
#         self.var = self.add_weight(name='var',
#                                    shape=input_shape[1:],
#                                    initializer='ones',
#                                    trainable=False)
#         super(Standardization, self).build(input_shape)
#     def call(self, x):
#         self.mean = K.mean(x, axis=0)
#         self.var = K.var(x, axis=0)
#         y = (x - self.mean) / (K.sqrt(self.var) + K.epsilon())
#         return y
#     def compute_output_shape(self, input_shape):
#         return input_shape


# class BatchNormalization(Layer):
#     def __init__(self, **kwargs):
#         super(BatchNormalization, self).__init__(**kwargs)
#     def build(self,input_shape):
#         self.mean = self.add_weight(name='mean',
#                                     shape=input_shape[1:],
#                                     initializer='zeros',
#                                     trainable=False)
#         self.var = self.add_weight(name='var',
#                                    shape=input_shape[1:],
#                                    initializer='ones',
#                                    trainable=False)
#         self.gamma = self.add_weight(name='gamma',
#                                      shape=input_shape[1:],
#                                      initializer='ones',
#                                      trainable=True)
#         self.beta = self.add_weight(name='beta',
#                                     shape=input_shape[1:],
#                                     initializer='zeros',
#                                     trainable=True)
#         super(BatchNorm,self).build(input_shape)
#     def call(self, x):
#         self.mean, self.var = K.mean(x, axis=0), K.var(x, axis=0)
#         y = (x - self.mean) / (K.sqrt(self.var) + K.epsilon())
#         y = self.gamma * y + self.beta
#         return y
#     def compute_output_shape(self,input_shape):
#         return input_shape


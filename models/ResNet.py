# -*- coding: utf-8 -*-

from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D


"""
TODO:
__init__ のパラメータ調整
ショートカットコネクションのfilter数, kernel_size, stridesを入力と比較して, なんとかする
"""

def add(**kwargs):
    return Add(**kwargs)


def batch_normalization(axis=-1, **kwargs):
    return BatchNormalization(axis=axis, **kwargs)


def conv2d(filters, kernel_size, strides=1, **kwargs):
    return Conv2D(filters,
                  kernel_size,
                  strides=strides,
                  padding='same',
                  use_bias=False,
                  kernel_initializer=initializers.he_normal(),
                  kernel_regularizer=regularizers.l2(0.0001),
                  **kwargs)


def dense(units, **kwargs):
    return Dense(units,
                 kernel_regularizer=regularizers.l2(0.0001),
                 bias_regularizer=regularizers.l2(0.0001),
                 **kwargs)


def global_average_pooling2d(**kwargs):
    return GlobalAveragePooling2D(**kwargs)


def relu(**kwargs):
    return Activation(activations.relu)


def softmax(axis=-1, **kwargs):
    return Activation(activations.softmax)


class ResNet:
    """
    Residual Network for Cifar10
    
    References
    - [Deep Residual Learning for Image Recognition] (http://arxiv.org/abs/1512.03385)
    - [Identity Mappings in Deep Residual Networks] (https://arxiv.org/abs/1603.05027)
    """

    """
    parameters
    n?
    blocks = [3, 3, 3] or [2, 2, 2, 2]
    layers: num of all weighted layer
    filters: filter list or initial filter

    blocks is calced with n and layers.
    complete blocks is needed. but n and layers is not needed.
    n, layers => one block has (layers - 2) / (2 n) conv (not bottleneck) 
                               (layers - 2) / (3 n) (bottleneck)
    """
    
    def __init__(self, blocks, filters, kernel_size=(3, 3), bottleneck=False):
        self.blocks = blocks
        if type(filters) == list:
            self.filters = filters
        else:
            self.filters = [pow(2, i) * filters for i in range(len(blocks))]
        self.kernel_size = kernel_size
        self.bottleneck = bottleneck

    def residual(self, inputs, filters, kernel_size=(3, 3), down_sampling=False):
        strides = 2 if down_sampling else 1
        
        outputs = batch_normalization()(inputs)
        outputs = relu()(outputs)
        outputs = conv2d(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides)(outputs)
        if down_sampling:
            inputs = conv2d(filters=filters,
                            kernel_size=(1, 1),
                            strides=strides)(inputs)
        outputs = batch_normalization()(outputs)
        outputs = relu()(outputs)
        outputs = conv2d(filters=filters, kernel_size=kernel_size)(outputs)
        outputs = add()([inputs, outputs])

        return outputs

    def build(self, input_shape, classes=10):
        inputs = keras.Input(shape=input_shape)

        outputs = conv2d(self.filters[0],
                         kernel_size=self.kernel_size)(inputs)

        for b in range(len(self.blocks)):
            for l in range(self.blocks[b]):
                if b != 0 and l == 0:
                    outputs = self.residual(outputs,
                                            self.filters[b],
                                            down_sampling=True)
                else:
                    outputs = self.residual(outputs,
                                            self.filters[b])

        outputs = global_average_pooling2d()(outputs)
        outputs = dense(classes)(outputs)
        outputs = softmax()(outputs)

        model = keras.Model(inputs, outputs)

        model.summary()

        return model


if __name__ == '__main__':
    model = ResNet([3, 3, 3], [16, 32, 64]).build(input_shape=(32, 32, 3), classes=10)

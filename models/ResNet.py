# -*- coding: utf-8 -*-

from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D


def add(**kwargs):
    return Add(**kwargs)


def average_pooling2d(pool_size=(2, 2), strides=None, **kwargs):
    return AveragePooling2D(pool_size=pool_size,
                            strides=strides,
                            padding='same',
                            **kwargs)


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


def max_pooling2d(pool_size=(2, 2), strides=None, **kwargs):
    return MaxPooling2D(pool_size=pool_size,
                        strides=strides,
                        padding='same',
                        **kwargs)


def relu(**kwargs):
    return Activation(activation=activations.relu, **kwargs)


def softmax(axis=-1, **kwargs):
    return Activation(activation=activations.softmax, **kwargs)


class ResNet:
    """
    Residual Network for Cifar10
    
    References
    - [Deep Residual Learning for Image Recognition] (http://arxiv.org/abs/1512.03385)
    - [Identity Mappings in Deep Residual Networks] (https://arxiv.org/abs/1603.05027)
    """
    def __init__(self, blocks, filters, kernel_size=(3, 3), bottleneck=False):
        if type(blocks) == list:
            self.blocks = blocks
        else:
            self.blocks = [blocks for _ in range(3)]
        if type(filters) == list:
            self.filters = filters
        else:
            self.filters = [pow(2, i) * filters for i in range(len(self.blocks))]
        self.kernel_size = kernel_size
        self.bottleneck = bottleneck
        self.bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
        self.imagenet = False

    def residual(self, inputs, filters, sub_sampling=False):
        strides = 2 if sub_sampling else 1
        kernel_size = (1, 1) if self.bottleneck else (3, 3)

        outputs = batch_normalization()(inputs)
        outputs = relu()(outputs)
        outputs = conv2d(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides
                         )(outputs)
        if sub_sampling or self.bottleneck:
            inputs = conv2d(filters=4 * filters if self.bottleneck else filters,
                            kernel_size=(1, 1),
                            strides=strides
                            )(inputs)
        outputs = batch_normalization()(outputs)
        outputs = relu()(outputs)
        outputs = conv2d(filters=filters,
                         kernel_size=(3, 3),
                         strides=1
                         )(outputs)
        if self.bottleneck:
            outputs = batch_normalization()(outputs)
            outputs = relu()(outputs)
            outputs = conv2d(filters=4*filters,
                             kernel_size=kernel_size,
                             strides=1
                             )(outputs)
        outputs = add()([inputs, outputs])

        return outputs

    def build(self, input_shape, classes=10):
        inputs = keras.Input(shape=input_shape)

        if self.imagenet:
            outputs = conv2d(filters=self.filters[0],
                             kernel_size=(7, 7),
                             strides=2
                             )(inputs)
            outputs = max_pooling2d(pool_size=(3, 3), strides=2)(outputs)
        else:
            outputs = conv2d(self.filters[0],
                             kernel_size=self.kernel_size
                             )(inputs)

        for b in range(len(self.blocks)):
            for l in range(self.blocks[b]):
                sub_sampling = True if b != 0 and l == 0 else False
                outputs = self.residual(outputs,
                                        self.filters[b],
                                        sub_sampling=sub_sampling)

        if self.imagenet:
            outputs = average_pooling2d(pool_size=(2, 2))(outputs)
        else:
            outputs = global_average_pooling2d()(outputs)
        outputs = dense(classes)(outputs)
        outputs = softmax()(outputs)

        model = keras.Model(inputs, outputs)

        model.summary()

        return model


def ResNetB(blocks, filters, kernel_size=(3, 3)):
    return ResNet(blocks=blocks, filters=filters, kernel_size=kernel_size, bottleneck=True)


def ResNet20():
    return ResNet(blocks=3, filters=16)


def ResNet32():
    return ResNet(blocks=5, filters=16)


def ResNet44():
    return ResNet(blocks=7, filters=16)


def ResNet56():
    return ResNet(blocks=9, filters=16)


def ResNet110():
    return ResNet(blocks=18, filters=16)


def ResNet29():
    return ResNet(blocks=3, filters=16, bottleneck=True)


def ResNet47():
    return ResNet(blocks=5, filters=16, bottleneck=True)


def ResNet65():
    return ResNet(blocks=7, filters=16, bottleneck=True)


def ResNet83():
    return ResNet(blocks=9, filters=16, bottleneck=True)


def ResNet164():
    return ResNet(blocks=18, filters=16, bottleneck=True)

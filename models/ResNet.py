# -*- coding: utf-8 -*-

from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import Conv2D, Dense

from .__layers__ import (add, average_pooling2d, batch_normalization, flatten,
                         global_average_pooling2d, max_pooling2d, relu,
                         softmax)


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


def residual(inputs, filters, bottleneck=False, sub_sampling=False):
    strides = 2 if sub_sampling else 1
    kernel_size = (1, 1) if bottleneck else (3, 3)

    outputs = batch_normalization()(inputs)
    outputs = relu()(outputs)
    outputs = conv2d(filters=filters, kernel_size=kernel_size,
                     strides=strides)(outputs)
    if sub_sampling or bottleneck:
        inputs = conv2d(filters=4 * filters if bottleneck else filters,
                        kernel_size=(1, 1),
                        strides=strides)(inputs)
    outputs = batch_normalization()(outputs)
    outputs = relu()(outputs)
    outputs = conv2d(filters=filters, kernel_size=(3, 3), strides=1)(outputs)
    if bottleneck:
        outputs = batch_normalization()(outputs)
        outputs = relu()(outputs)
        outputs = conv2d(filters=4*filters,
                         kernel_size=kernel_size, strides=1)(outputs)
    outputs = add()([inputs, outputs])

    return outputs


class ResNet:

    """
    Residual Network

    Parameter:
    - blocks: num of blocks as list
    - filters: num of filter map
    - bottleneck: If True use bottleneck architecture
    - input_layers: input layers for network
        for ImageNet: input_layers=[conv2d(filters=64, kernel_size=(7, 7), strides=2)] (default)
        for Cifar: input_layers=[conv2d(filters=16, kernel_size=(3, 3))]
    - output_layers: output layers for network
        for ImageNet: output_layers=[average_pooling2d(pool_size=(2, 2)), flatten()])
        for Cifar: output_layers=[global_average_pooling2d()]

    References
    - [Deep Residual Learning for Image Recognition] (http://arxiv.org/abs/1512.03385)
    - [Identity Mappings in Deep Residual Networks] (https://arxiv.org/abs/1603.05027)
    """

    def __init__(self, blocks, filters, bottleneck=False,
                 input_layers=[
                     conv2d(filters=64, kernel_size=(7, 7), strides=2),
                     batch_normalization(),
                     relu()
                 ],
                 output_layers=[average_pooling2d(pool_size=(2, 2)), flatten()]):
        self.blocks = blocks
        self.filters = filters
        self.bottleneck = bottleneck
        self.bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
        self.input_layers = input_layers
        self.output_layer = output_layers

    def build(self, input_shape, classes=1000):
        inputs = keras.Input(shape=input_shape)

        outputs = inputs
        for layer in self.input_layers:
            outputs = layer(outputs)

        for b in range(len(self.blocks)):
            for l in range(self.blocks[b]):
                sub_sampling = True if b != 0 and l == 0 else False
                outputs = residual(outputs, self.filters[b],
                                   bottleneck=self.bottleneck, sub_sampling=sub_sampling)

        for layer in self.output_layer:
            outputs = layer(outputs)

        outputs = dense(classes)(outputs)
        outputs = softmax()(outputs)

        model = keras.Model(inputs, outputs)

        model.summary()

        return model


def ResNet18():
    """
    ResNet for ImageNet with 18 layers.
    """
    return ResNet(blocks=[2, 2, 2, 2], filters=[64, 128, 256, 512])


def ResNet34():
    """
    ResNet for ImageNet with 34 layers.
    """
    return ResNet(blocks=[3, 4,  6, 3], filters=[64, 128, 256, 512])


def ResNet50():
    """
    ResNet for ImageNet with 50 layers.
    """
    return ResNet(blocks=[3, 4, 6, 3], filters=[64, 128, 256, 512], bottleneck=True)


def ResNet101():
    """
    ResNet for ImageNet with 101 layers.
    """
    return ResNet(blocks=[3, 4, 23, 3], filters=[64, 128, 256, 512], bottleneck=True)


def ResNet152():
    """
    ResNet for ImageNet with 152 layers.
    """
    return ResNet(blocks=[3, 4, 36, 3], filters=[64, 128, 256, 512], bottleneck=True)


def ResNet20():
    """
    ResNet for Cifar10/100 with 20 layers.
    """
    return ResNet(blocks=[3, 3, 3], filters=[16, 32, 64], input_layers=[conv2d(filters=16, kernel_size=(3, 3))], output_layers=[global_average_pooling2d()])


def ResNet32():
    """
    ResNet for Cifar10/100 with 32 layers.
    """
    return ResNet(blocks=[5, 5, 5], filters=[16, 32, 64], input_layers=[conv2d(filters=16, kernel_size=(3, 3))], output_layers=[global_average_pooling2d()])


def ResNet44():
    """
    ResNet for Cifar10/100 with 44 layers.
    """
    return ResNet(blocks=[7, 7, 7], filters=[16, 32, 64], input_layers=[conv2d(filters=16, kernel_size=(3, 3))], output_layers=[global_average_pooling2d()])


def ResNet56():
    """
    ResNet for Cifar10/100 with 56 layers.
    """
    return ResNet(blocks=[9, 9, 9], filters=[16, 32, 64], input_layers=[conv2d(filters=16, kernel_size=(3, 3))], output_layers=[global_average_pooling2d()])

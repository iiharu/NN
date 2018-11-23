# -*- coding: utf-8 -*-

from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import Conv2D, Dense

from .__layers__ import (add, average_pooling2d, batch_normalization, dropout, flatten,
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
                     batch_normalization(),
                     relu(),
                     conv2d(filters=64, kernel_size=(7, 7), strides=2),
                 ],
                 output_layers=[average_pooling2d(pool_size=(2, 2)), flatten()]):
        self.blocks = blocks
        self.filters = filters
        self.bottleneck = bottleneck
        self.bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
        self.input_layers = input_layers
        self.output_layer = output_layers

    def residual(self, inputs, filters, bottleneck=False, sub_sampling=False):
        strides = 2 if sub_sampling else 1
        kernel_size = (1, 1) if bottleneck else (3, 3)

        outputs = batch_normalization()(inputs)
        outputs = relu()(outputs)
        outputs = conv2d(filters=filters, kernel_size=kernel_size,
                         strides=strides)(outputs)
        if sub_sampling or bottleneck:
            # inputs = batch_normalization()(inputs)
            # inputs = relu()(inputs)
            inputs = conv2d(filters=4 * filters if bottleneck else filters,
                            kernel_size=(1, 1),
                            strides=strides)(inputs)
        outputs = batch_normalization()(outputs)
        outputs = relu()(outputs)
        outputs = conv2d(filters=filters, kernel_size=(
            3, 3), strides=1)(outputs)
        if bottleneck:
            outputs = batch_normalization()(outputs)
            outputs = relu()(outputs)
            outputs = conv2d(filters=4*filters,
                             kernel_size=kernel_size, strides=1)(outputs)
        outputs = add()([inputs, outputs])

        return outputs

    def build(self, input_shape, classes=1000):
        inputs = keras.Input(shape=input_shape)

        outputs = inputs
        for layer in self.input_layers:
            outputs = layer(outputs)

        for b in range(len(self.blocks)):
            for l in range(self.blocks[b]):
                sub_sampling = True if b != 0 and l == 0 else False
                outputs = self.residual(outputs, self.filters[b],
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


class WideResNet:

    """
    Wide Residual Networks

    Parameter:

    References:
    - [Wide Residual Networks] (https://arxiv.org/abs/1605.07146)
    """

    def __init__(self, layers, widening, deeping=2, dropout_rate=0.0):
        self.layers = layers  # n: num of convlutions
        self.widening = widening  # k: widening factor
        self.deeping = deeping  # l: deeping factor
        # kernel_size in residual block
        self.block_type = [(3, 3)] * self.deeping
        # d: num of residual blocks in conv_i
        self.blocks = (layers - 4) // (3 * self.deeping)
        self.dropout_rate = dropout_rate
        self.bn_axis = -1 if K.image_data_format() == 'channels_last' else 1  # feature map axis

    def conv(self, inputs, filters, kernel_size, strides=1):
        inputs = batch_normalization()(inputs)
        inputs = relu()(inputs)
        inputs = conv2d(filters=filters, kernel_size=kernel_size,
                        strides=strides)(inputs)
        return inputs

    def residual(self, inputs, filters, down_sampling=False):

        outputs = inputs
        for i, block in enumerate(self.block_type):
            strides = 2 if down_sampling and i == 0 else 1
            if i > 0 and self.dropout_rate > 0.0:
                outputs = dropout(rate=self.dropout_rate)(outputs)
            outputs = self.conv(outputs, filters=filters,
                                kernel_size=block, strides=strides)

        if K.int_shape(outputs)[self.bn_axis] != K.int_shape(inputs)[self.bn_axis]:
            strides = 2 if down_sampling else 1
            inputs = self.conv(inputs, filters=filters,
                               kernel_size=(1, 1), strides=strides)
        outputs = add()([inputs, outputs])

        return outputs

    def build(self, input_shape, classes=10):

        filters = 16

        inputs = keras.Input(shape=input_shape)

        outputs = batch_normalization()(inputs)
        outputs = relu()(outputs)
        outputs = conv2d(filters=filters, kernel_size=(3, 3))(outputs)

        filters = filters * self.widening

        for i in range(3):
            for j in range(self.blocks):
                down_sampling = True if (i > 0 and j == 0) else False
                outputs = self.residual(
                    outputs, filters=filters, down_sampling=down_sampling)

            filters = 2 * filters

        outputs = batch_normalization()(outputs)
        outputs = relu()(outputs)
        outputs = average_pooling2d(pool_size=(8, 8))(outputs)
        outputs = dense(10)(outputs)
        outputs = softmax()(outputs)

        model = keras.Model(inputs, outputs)

        model.summary()

        return model


def WideResNetD40K4():
    return WideResNet(layers=40, widening=4)


def WideResNetD16K8():
    return WideResNet(layers=16, widening=8)


def WideResNetD28K10():
    return WideResNet(layers=28, widening=10)


def WideResNetD28K10D(dropout_rate=0.3):
    return WideResNet(layers=28, widening=10, dropout_rate=0.3)

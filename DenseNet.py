# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras

from layers import average_pooling2d, batch_normalization, concat, global_average_pooling2d, relu, softmax


def conv2d(filters, kernel_size, strides=1, **kwargs):
    return keras.layers.Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same',
                         use_bias=False,
                         kernel_initializer=keras.initializers.he_normal(),
                         kernel_regularizer=keras.regularizers.l2(0.0001),
                         **kwargs)


def dense(units, **kwargs):
    return keras.layers.Dense(units,
                        kernel_regularizer=keras.regularizers.l2(0.0001),
                        bias_regularizer=keras.regularizers.l2(0.0001),
                        **kwargs)


class DenseNet:
    """

    DenseNet.
    This model is tested with Cifar10/Cifar100.

    Reference:
    - Densely Connected Convolutional Networks
    - https://github.com/liuzhuang13/DenseNet

    Variance:
    - DenseNet
      - Layers: 40, GrowthRate: 12
      - Layers: 100, GrowthRate: 12
      - Layers: 100, GrowthRate: 24
    - DenseNetBC
      - Layers: 100, GrowthRate: 12
      - Layers: 250, GrowthRate: 24
      - Layers: 190, GrowthRate: 40

    Examples:
    model = DenseNet(layers=40, growth_rate=12, blocks=3).build(input_shape=(32, 32, 3), classes=10)
    """

    def __init__(self, layers, growth_rate, blocks,
                 bottleneck=False, compression=False, reduction_rate=0.5):
        self.layers = layers
        self.growth_rate = growth_rate
        if type(blocks) == list:
            self.blocks = blocks
        else:
            convs = (self.layers - blocks - 1) // blocks
            if bottleneck:
                convs = convs // 2
            self.blocks = [convs for i in range(blocks)]
        self.bottleneck = bottleneck
        self.compression = compression
        self.reduction_rate = reduction_rate
        self.batch_normalization_axis = - \
            1 if keras.backend.image_data_format() == 'channels_last' else 1

    def conv_block(self, inputs):
        outputs = batch_normalization()(inputs)
        if self.bottleneck:
            outputs = relu()(outputs)
            outputs = conv2d(filters=4 * self.growth_rate,
                             kernel_size=(1, 1))(outputs)
            outputs = batch_normalization()(outputs)
        outputs = relu()(outputs)
        outputs = conv2d(filters=self.growth_rate,
                         kernel_size=(3, 3))(outputs)
        outputs = concat()([inputs, outputs])

        return outputs

    def dense_block(self, inputs, convs):
        for _ in range(convs):
            inputs = self.conv_block(inputs)
        return inputs

    def transition_block(self, inputs):
        filters = keras.backend.int_shape(inputs)[self.batch_normalization_axis]
        if self.compression:
            filters = int(filters * (1 - self.reduction_rate))

        outputs = batch_normalization()(inputs)
        outputs = relu()(outputs)
        outputs = conv2d(filters=filters,
                         kernel_size=(1, 1))(outputs)
        outputs = average_pooling2d(pool_size=(2, 2),
                                    strides=2)(outputs)

        return outputs

    def build(self, input_shape, classes):
        inputs = keras.Input(shape=input_shape)

        if self.bottleneck and self.compression:
            outputs = conv2d(filters=2 * self.growth_rate,
                             kernel_size=(3, 3))(inputs)
        else:
            outputs = conv2d(filters=16, kernel_size=(3, 3))(inputs)

        for i in range(len(self.blocks)):
            outputs = self.dense_block(outputs,
                                       self.blocks[i])
            if i < len(self.blocks) - 1:
                outputs = self.transition_block(outputs)

        outputs = global_average_pooling2d()(outputs)
        outputs = dense(classes)(outputs)
        outputs = softmax()(outputs)

        model = keras.Model(inputs, outputs)

        model.summary()

        return model


def DenseNetBC(layers, growth_rate, blocks, reduction_rate=0.5):
    return DenseNet(layers=layers,
                    growth_rate=growth_rate,
                    blocks=blocks,
                    bottleneck=True,
                    compression=True,
                    reduction_rate=reduction_rate)


# def DenseNetL40K12():
#	 return DenseNet(layers=40, growth_rate=12, blocks=3)
#
#
# def DenseNetL100K12():
#	 return DenseNet(layers=100, growth_rate=12, blocks=3)
#
#
# def DenseNetL100K24():
#	 return DenseNet(layers=100, growth_rate=24, blocks=3)
#
#
# def DenseNetBCL100K12():
#	 return DenseNetBC(layers=100, growth_rate=12, blocks=3)
#
#
# def DenseNetBCL250K24():
#	 return DenseNetBC(layers=250, growth_rate=24, blocks=3)
#
#
# def DenseNetBCL190K40():
#	 return DenseNetBC(layers=190, growth_rate=40, blocks=3)
#
#

# def DenseNet121():
#   return DenseNet(blocks=[6, 12, 24, 16], growth_rate=32, bottleneck=True)
# def DenseNet169():
#   return DenseNet(blocks=[6, 12, 32, 32], growth_rate=32)
# def DenseNet201():
#   return DenseNet(blocks=[6, 12, 48, 32], growth_rate=32)
# def DenseNet264():
#   return DenseNet(blocks=[6, 12, 64, 48], growth_rate=32)


if __name__ == '__main__':
    model = DenseNet(layers=40, growth_rate=12, blocks=3).build(
        input_shape=(32, 32, 3), classes=10)

# -*- coding: utf-8 -*-


from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import Conv2D, Dense

from .__layers__ import (average_pooling2d, batch_normalization,
                         concat, global_average_pooling2d, relu, softmax)


def conv2d(filters, kernel_size, strides=1, **kwargs):
    return Conv2D(filters=filters,
                  kernel_size=kernel_size,
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
            1 if K.image_data_format() == 'channels_last' else 1

    def conv_block(self, inputs, name):
        outputs = batch_normalization(name=name + "/" + "BN")(inputs)
        if self.bottleneck:
            outputs = relu(name=name + "/" + "ReLu")(outputs)
            outputs = conv2d(filters=4 * self.growth_rate,
                             kernel_size=(1, 1),
                             name=name + "/" + "Conv")(outputs)
            outputs = batch_normalization(name=name + "/" + "BN")(outputs)
        outputs = relu(name=name + "/" + "ReLu")(outputs)
        outputs = conv2d(filters=self.growth_rate,
                         kernel_size=(3, 3),
                         name=name + "/" + "Conv")(outputs)
        outputs = concat(name=name + "/" + "Concat")([inputs, outputs])

        return outputs

    def dense_block(self, inputs, convs, name):
        for i in range(convs):
            inputs = self.conv_block(inputs, name=name + "/" + "Conv" + str(i))
        return inputs

    def transition_block(self, inputs, name):
        filters = K.int_shape(inputs)[self.batch_normalization_axis]
        if self.compression:
            filters = int(filters * (1 - self.reduction_rate))

        outputs = batch_normalization(name=name + "/" + "BN")(inputs)
        outputs = relu(name=name + "/" + "ReLU")(outputs)
        outputs = conv2d(filters=filters,
                         kernel_size=(1, 1),
                         name=name + "/" + "Conv"
                         )(outputs)
        outputs = average_pooling2d(pool_size=(2, 2),
                                    strides=2,
                                    name=name + "/" + "AvePool"
                                    )(outputs)

        return outputs

    def build(self, input_shape, classes, name="DenseNet"):
        inputs = keras.Input(shape=input_shape, name=name + "/" + "Input")

        if self.bottleneck and self.compression:
            outputs = conv2d(filters=2 * self.growth_rate,
                             kernel_size=(3, 3),
                             name="Conv"
                             )(inputs)
        else:
            outputs = conv2d(filters=16, kernel_size=(
                3, 3), name="Conv")(inputs)

        for i in range(len(self.blocks)):
            outputs = self.dense_block(outputs,
                                       self.blocks[i],
                                       name="Dense" + str(i))
            if i < len(self.blocks) - 1:
                outputs = self.transition_block(outputs,
                                                name="Transition" + str(i))

        outputs = global_average_pooling2d(name="GlobalAvePool")(outputs)
        outputs = dense(classes, name="FullyConnect")(outputs)
        outputs = softmax(name="SoftMax")(outputs)

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
if __name__ == '__main__':
    model = DenseNet(layers=40, growth_rate=12, blocks=3).build(
        input_shape=(32, 32, 3), classes=10)

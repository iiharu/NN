# -*- coding: utf-8 -*-

from tensorflow import keras
from tensorflow.keras import backend as K

from .__layers__ import (average_pooling2d, batch_normalization, concat,
                         conv2d, dense, dropout, flatten, max_pooling2d, relu, softmax)


class GoogLeNet:
    """
    GoogLeNet (Inception v1)

    If use another inception such as v2 (without another split),
    use `inception' with layers parameter.

    - References:
      - [Going Deeper with Convolutions] (https://arxiv.org/abs/1409.4842v1)
      - [Rethinking the Inception Architecture for Computer Vision] (https://arxiv.org/abs/1512.00567v3)
      - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning] (https://arxiv.org/abs/1602.07261)

    - Examples:
    model = GoogLeNet().build(input_shape=(224, 224, 3), classes=1000)
    model.compile(optimizer=optimizer,
                  loss=[losses.categorical_crossentropy,
                        losses.categorical_crossentropy,
                        losses.categorical_crossentropy],
                  loss_weights=[1.0, 0.3, 0.3],
                  metrics=[metrics.categorical_accuracy,
                           metrics.top_k_categorical_accuracy])
    """

    def __init__(self, input_layers=None, outputs_layers=None, inception_layers=None):
        self.inception_version = 1
        if input_layers is None:
            self.input_layers = [conv2d(filters=64, kernel_size=(7, 7),
                                        strides=2, padding='same'),
                                 max_pooling2d(pool_size=(3, 3),
                                               strides=2, padding='same'),
                                 batch_normalization(),
                                 conv2d(filters=192, kernel_size=(1, 1),
                                        strides=1, padding='valid'),
                                 conv2d(filters=192, kernel_size=(3, 3),
                                        strides=1, padding='same'),
                                 batch_normalization(),
                                 max_pooling2d(pool_size=(3, 3),
                                               strides=2, padding='same')]
        else:
            self.input_layers = input_layers

        if outputs_layers is None:
            self.output_layers = [average_pooling2d(pool_size=(7, 7),
                                                    strides=1, padding='valid'),
                                  flatten(),
                                  dropout(0.4),
                                  dense(1000),
                                  softmax()]
        else:
            self.output_layers = outputs_layers

    def inception(self, inputs, filters, layers=None):
        """
        Inception Module (default is inception v1)

        If you want to use another inception module,
        set `layers' appropriate layer function.
        Below is inception v2 (Module A) example:
        inception_layers = [[conv2d(filters=filters, kernel_size=(1, 1))],
                            [conv2d(filters=filters, kernel_size=(1, 1)),
                             conv2d(filters=filters, kernel_size=(3, 3))],
                            [conv2d(filters=filters, kernel_size=(1, 1)),
                             conv2d(filters=filters, kernel_size=(3, 3))
                             conv2d(filters=filters, kernel_size=(3, 3))],
                            [max_pooling2d(pool_size=(3, 3), strides=1),
                             conv2d(filters=filters, kernel_size=(1, 1))]]
        outputs = self.inception(inputs, filters, layers=inception_layers)
        """
        if layers is None:
            layers = [[conv2d(filters=filters, kernel_size=(1, 1))],
                      [conv2d(filters=filters, kernel_size=(1, 1)),
                       conv2d(filters=filters, kernel_size=(3, 3))],
                      [conv2d(filters=filters, kernel_size=(1, 1)),
                       conv2d(filters=filters, kernel_size=(5, 5))],
                      [max_pooling2d(pool_size=(3, 3), strides=1),
                       conv2d(filters=filters, kernel_size=(1, 1))]]

        outputs = [inputs] * len(layers)

        for i in range(len(layers)):
            for layer in layers[i]:
                outputs[i] = layer(outputs[i])

        outputs = concat()(outputs)

        return outputs

    def classifier_main(self, inputs, classes):
        outputs = average_pooling2d(pool_size=(7, 7),
                                    strides=1, padding='valid')(inputs)
        outputs = flatten()(outputs)
        outputs = dropout(0.4)(outputs)
        outputs = dense(classes)(outputs)
        outputs = softmax()(outputs)

        return outputs

    def classifier_aux(self, inputs, classes):
        filters = 128
        outputs = average_pooling2d(pool_size=(5, 5),
                                    strides=3, padding='valid')(inputs)
        outputs = conv2d(filters=filters, kernel_size=(1, 1),
                         strides=1, padding='same')(outputs)
        outputs = flatten()(outputs)
        outputs = relu()(outputs)
        outputs = dense(1024)(outputs)
        outputs = relu()(outputs)
        outputs = dropout(0.7)(outputs)
        outputs = dense(classes)(outputs)
        outputs = softmax()(outputs)

        return outputs

    def build(self, input_shape, classes):
        # input
        inputs = keras.Input(shape=input_shape)

        filters = 64
        outputs = conv2d(filters=filters, kernel_size=(7, 7),
                         strides=2, padding='same')(inputs)
        outputs = max_pooling2d(pool_size=(3, 3),
                                strides=2, padding='same')(outputs)
        filters = 192
        # using batch norm instead of local response norm
        outputs = batch_normalization()(outputs)
        outputs = conv2d(filters=filters, kernel_size=(1, 1),
                         strides=1, padding='valid')(outputs)
        outputs = conv2d(filters=filters, kernel_size=(3, 3),
                         strides=1, padding='same')(outputs)
        # using batch norm instead of local response norm
        outputs = batch_normalization()(outputs)
        outputs = max_pooling2d(pool_size=(3, 3),
                                strides=2,  padding='same')(outputs)

        # inception (3a)
        filters = 256
        outputs = self.inception(inputs=outputs, filters=filters)
        # inception (3b)
        filters = 480
        outputs = self.inception(inputs=outputs, filters=filters)

        outputs = max_pooling2d(pool_size=(3, 3),
                                strides=2, padding='same')(outputs)

        # inception (4a)
        filters = 512
        outputs = self.inception(inputs=outputs, filters=filters)

        # if K.learning_phase() == 1:
        outputs2 = self.classifier_aux(outputs, classes=classes)

        # inception (4b)
        outputs = self.inception(inputs=outputs, filters=filters)
        # inception (4c)
        outputs = self.inception(inputs=outputs, filters=filters)
        # inception (4d)
        filters = 528
        outputs = self.inception(inputs=outputs, filters=filters)

        # if K.learning_phase() == 1:
        outputs1 = self.classifier_aux(outputs, classes=classes)

        # inception (4e)
        filters = 832
        outputs = self.inception(inputs=outputs, filters=filters)

        outputs = max_pooling2d(pool_size=(2, 2),
                                strides=2, padding='same')(outputs)

        # inception (5a)
        outputs = self.inception(inputs=outputs, filters=filters)
        # inception (5b)
        filters = 1024
        outputs = self.inception(inputs=outputs, filters=filters)

        # classifier
        outputs0 = self.classifier_main(inputs=outputs, classes=1000)

        model = keras.Model(inputs=inputs,
                            outputs=[outputs0, outputs1, outputs2])

        model.summary()

        return model

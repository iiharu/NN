# -*- coding: utf-8 -*-

from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

from .__layers__ import (average_pooling2d, batch_normalization, concat,
                         conv2d, dense, dropout, flatten, max_pooling2d, relu, softmax)

# def CONV(inputs, filters, kernel_size, strides=1):
#     outputs = batch_normalization()(inputs)
#     outputs = relu()(outputs)
#     outputs = conv2d(filters=filters, kernel_size=kernel_size,
#                      strides=strides)(inputs)
#     return outputs


# def conv_block(inputs, filters, kernel_size, strides=1, activation=None, use_bias=True,
#                kernel_initializer='glorot_uniform', bias_initializer='zeros',
#                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#                kernel_constraint=None, bias_constraint=None):
#     inputs = batch_normalization()(inputs)
#     inputs = relu()(inputs)
#     inputs = conv2d(filters=filters, kernel_size=kernel_size, strides=strides, activation=activation, use_bias=True,
#                     kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
#                     kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
#                     kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(inputs)
#     return inputs


# def fc_softmax(inputs, classes, use_bias=True,
#                kernel_initializer='glorot_uniform', bias_initializer='zeros',
#                kernel_regularizer=None, bias_regularizer=None,
#                activity_regularizer=None,
#                kernel_constraint=None, bias_constraint=None):
#     inputs = dense(classes, use_bias=use_bias,
#                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
#                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
#                    activity_regularizer=activity_regularizer,
#                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(inputs)
#     inputs = softmax()(inputs)
#     return inputs


class GoogLeNet:
    """
    GoogLeNet

    - References:

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

    def __init__(self):
        self.block = 2

    def inception(self, inputs, filters):
        outputs1 = inputs
        outputs2 = inputs
        outputs3 = inputs
        outputs4 = inputs

        outputs1 = conv2d(filters=filters,
                          kernel_size=(1, 1), strides=1)(outputs1)

        outputs2 = conv2d(filters=filters,
                          kernel_size=(1, 1), strides=1)(outputs2)
        outputs2 = conv2d(filters=filters,
                          kernel_size=(3, 3), strides=1)(outputs2)

        outputs3 = conv2d(filters=filters,
                          kernel_size=(1, 1), strides=1)(outputs3)
        outputs3 = conv2d(filters=filters,
                          kernel_size=(5, 5), strides=1)(outputs3)

        outputs4 = max_pooling2d(pool_size=(3, 3), strides=1)(outputs4)
        outputs4 = conv2d(filters=filters,
                          kernel_size=(1, 1), strides=1)(outputs4)

        outputs = concat()([outputs1, outputs2, outputs3, outputs4])

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

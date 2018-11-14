# -*- encoding: utf-8 -*-

from tensorflow import keras
from tensorflow.keras import backend as K


def block(inputs, layers):
    """
    Block function to define layer block function.

    Example:
    def conv_block(inputs, filters, kernel_size=(3, 3), strides=1):
        outputs = block(inputs, layers=[batch_normalization(), 
                                        relu(),
                                        conv2d(filters=filters, 
                                               kernel_size=kernel_size, 
                                               strides=strides)])
        return outputs
    """
    if layers is None:
        layers = []
    for layer in layers:
        inputs = layer(inputs)
    return inputs


def residual_block(inputs, layers, merge, res_layers):
    """
    Residual block function.

    TODO:
    - use `block' as parameter?

    Example:
    outputs = residual_block(inputs, 
                             layers=[batch_normalization(), 
                                     relu(),
                                     conv2d(filters=64, 
                                            kernel_size=(3, 3), 
                                            strides=(2, 2))],
                             merge=add(),
                             res_layers=[batch_normalization(), 
                                         relu(),
                                         conv2d(filters=64, 
                                                kernel_size=(1, 1), 
                                                strides=(2, 2))])
    """
    outputs = block(inputs, layers)
    if res_layers is None:
        res_layers = []
    for layer in res_layers:
        inputs = layer(inputs)

    outputs = merge([inputs, outputs])

    return outputs

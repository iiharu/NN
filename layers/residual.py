# -*- coding: utf-8 -*-

"""
Residual Block Layer
"""

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec

from tensorflow.python.keras.utils import conv_utils


class Residual(Layer):
    """
    Residual Block Layer

    # Example

    # Arguments
        filters: output filters
        kernel_size:
        activation:
        use_bias


    # References
        - [Deep Residual Learning for Image Recognition] (http://arxiv.org/abs/1512.03385)
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = 'same'
        self.data_format = 'channels_last'
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim_1 = input_shape[channel_axis]
        kernel_shape_1 = self.kernel_size + (input_dim_1, self.filters)

        self.kernel_1 = self.add_weight(name='kernel_1',
                                        shape=kernel_shape_1,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias_1 = self.add_weight(name='bias_1', shape=(self.filters, ),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint)
        else:
            self.bias_1 = None

        input_dim_2 = self.filters
        kernel_shape_2 = self.kernel_size + (input_dim_2, self.filters)
        self.kernel_2 = self.add_weight(name='kernel_2',
                                        shape=kernel_shape_2,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias_2 = self.add_weight(name='bias_2',shape=(self.filters, ),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint)
        else:
            self.bias_2 = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim_1})
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], ) + tuple(new_space) + (self.filters, )
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
        return (input_shape[0], self.filters) + tuple(new_space)

    def call(self, inputs, **kwargs):
        outputs = K.conv2d(x=inputs,
                           kernel=self.kernel_1,
                           strides=2,
                           padding='same')
        if self.use_bias:
            outputs = K.bias_add(x=outputs,
                                 bias=self.bias_1)
        outputs = K.conv2d(x=outputs,
                           kernel=self.kernel_2,
                           padding='same')
        if self.use_bias:
            outputs = K.bias_add(x=outputs,
                                 bias=self.bias_2)
        if K.shape(inputs) != K.shape(outputs):
            K.reshape(inputs, K.shape(outputs))
        outputs = inputs + outputs
        return outputs

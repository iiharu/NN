# -*- coding: utf-8 -*-

from tensorflow import keras


def conv_lstm2d(filters, kernel_size, strides=(1, 1),
                activation='tanh', recurrent_activation='hard_sigmoid',
                use_bias=True,
                kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
                unit_forget_bias=True,
                kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                return_sequences=False, go_backwards=False, stateful=False,
                dropout=0., recurrent_dropout=0.):
    return keras.layers.ConvLSTM2D(filters, kernel_size, strides=strides,
                                   padding='same', data_format=None, dilation_rate=(1, 1),
                                   activation=activation, recurrent_activation=recurrent_activation,
                                   use_bias=use_bias,
                                   kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer,
                                   unit_forget_bias=unit_forget_bias,
                                   kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer,
                                   activity_regularizer=activity_regularizer,
                                   kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint,
                                   return_sequences=return_sequences, go_backwards=go_backwards, stateful=stateful,
                                   dropout=dropout, recurrent_dropout=recurrent_dropout)


def gru(units,
        activation='tanh', recurrent_activation='hard_sigmoid',
        use_bias=True,
        kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
        kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
        dropout=0.0, recurrent_dropout=0.0,
        implementation=1,
        return_sequences=False, return_state=False, go_backwards=False,
        stateful=False, unroll=False, reset_after=False):
    return keras.layers.GRU(units,
                            activation=activation, recurrent_activation=recurrent_activation,
                            use_bias=use_bias,
                            kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer,
                            activity_regularizer=activity_regularizer,
                            kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint,
                            dropout=dropout, recurrent_dropout=recurrent_dropout,
                            implementation=implementation,
                            return_sequences=return_sequences, return_state=return_state, go_backwards=go_backwards,
                            stateful=stateful, unroll=unroll, reset_after=reset_after)


def lstm(units,
         activation='tanh', recurrent_activation='hard_sigmoid',
         use_bias=True,
         kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
         unit_forget_bias=True,
         kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
         activity_regularizer=None,
         kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
         dropout=0.0, recurrent_dropout=0.0,
         implementation=1,
         return_sequences=False, return_state=False, go_backwards=False,
         stateful=False, unroll=False):
    return keras.layers.LSTM(units,
                             activation=activation, recurrent_activation=recurrent_activation,
                             use_bias=use_bias,
                             kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer,
                             unit_forget_bias=unit_forget_bias,
                             kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer,
                             activity_regularizer=activity_regularizer,
                             kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint,
                             dropout=dropout, recurrent_dropout=recurrent_dropout,
                             implementation=implementation,
                             return_sequences=return_sequences, return_state=return_state, go_backwards=go_backwards,
                             stateful=stateful, unroll=unroll)

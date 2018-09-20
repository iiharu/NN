# -*- coding: utf-8 -*-

from tensorflow import keras


def embedding(input_dim, output_dim,
              embeddings_initializer='uniform',
              embeddings_regularizer=None,
              activity_regularizer=None,
              embeddings_constraint=None,
              mask_zero=False,
              input_length=None,
              **kwargs):
    return keras.layers.Embedding(input_dim, output_dim,
                                  embeddings_initializer=embeddings_initializer,
                                  embeddings_regularizer=embeddings_regularizer,
                                  activity_regularizer=activity_regularizer,
                                  embeddings_constraint=embeddings_constraint,
                                  mask_zero=mask_zero,
                                  input_length=input_length,
                                  **kwargs)

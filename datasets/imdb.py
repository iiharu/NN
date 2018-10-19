# -*- coding: utf-8 -*-

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence


def load_data(num_words=None, maxlen=None):
	(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=num_words,
														  skip_top=0,
														  maxlen=maxlen,
														  seed=113,
														  start_char=1,
														  oov_char=2,
														  index_from=3)

	X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
	X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

	return (X_train, Y_train), (X_test, Y_test)

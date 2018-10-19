# -*- coding: utf-8 -*-

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical

CLASSES = 100
ROWS, COLS, CHS = 32, 32, 3

TRAIN_SIZE = 50000
TEST_SIZE = 10000


def load_data(label_mode="fine", normalize=True, flatten=False, one_hot_label=True):
	(X_train, Y_train), (X_test, Y_test) = cifar100.load_data(label_mode=label_mode)

	if normalize:
		X_train = X_train.astype(float)
		X_test = X_test.astype(float)
		X_train /= 255
		X_test /= 255

	if flatten:
		X_train = X_train.reshape((TRAIN_SIZE, ROWS * COLS * CHS))
		X_test = X_test.reshape((TEST_SIZE, ROWS * COLS * CHS))

	if one_hot_label:
		classes = 100 if label_mode == "fine" else 20
		Y_train = to_categorical(Y_train, classes)
		Y_test = to_categorical(Y_test, classes)

	return (X_train, Y_train), (X_test, Y_test)

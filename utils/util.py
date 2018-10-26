# coding: utf-8

from tensorflow.keras import backend as K

def bn_axis():
	return -1 if K.image_data_format() == 'channels_last' else 1


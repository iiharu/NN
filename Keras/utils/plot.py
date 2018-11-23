# -*- coding: utf-8 -*-

from matplotlib import pyplot


def plot(history, metrics=['loss']):
	"""
	Plot training and validation metrics (such as loss, accuracy).
	"""
	for metric in metrics:
		val_metric = "val" + "_" + metric
		pyplot.plot(history.history[metric])
		pyplot.plot(history.history[val_metric])
		pyplot.title(metric)
		pyplot.ylabel(metric)
		pyplot.xlabel('epoch')
		pyplot.legend(['train', 'test'], loc='best')
		pyplot.show()

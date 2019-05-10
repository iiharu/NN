# -*- coding: utf-8


import tensorflow as tf
from tensorflow import keras

from layers import concat, conv2d, dense, flatten, max_pooling2d, sigmoid, softmax
from datasets import mnist


class LeNet5:

    """
    LeNet-5

    Example:
    model = LeNet5().build()
    """

    def build(self, input_shape=(28, 28, 1), classes=10):
        inputs = keras.Input(shape=input_shape)

        outputs = conv2d(filters=6, kernel_size=(6, 6))(inputs)
        outputs = max_pooling2d(pool_size=(2, 2), strides=(2, 2))(outputs)
        outputs = sigmoid()(outputs)

        outputs = conv2d(filters=16, kernel_size=(6, 6))(inputs)
        outputs = max_pooling2d(pool_size=(2, 2), strides=(2, 2))(outputs)
        outputs = sigmoid()(outputs)

        outputs = flatten()(outputs)

        outputs = dense(120)(outputs)
        outputs = sigmoid()(outputs)

        outputs = dense(64)(outputs)
        outputs = sigmoid()(outputs)

        outputs = dense(classes)(outputs)
        outputs = softmax()(outputs)

        model = keras.Model(inputs, outputs)

        model.summary()

        return model


class LeNet:
    """
    LeNet-300-100

    Example:
    model = LeNet().build()
    """

    def build(self, input_shape=(28, 28, 1), classes=10):
        inputs = keras.Input(shape=input_shape)

        outputs = flatten()(inputs)
        outputs = dense(300)(outputs)
        outputs = sigmoid()(outputs)

        outputs = dense(100)(outputs)
        outputs = sigmoid()(outputs)

        outputs = dense(10)(outputs)
        outputs = softmax()(outputs)

        model = keras.Model(inputs, outputs)

        model.summary()

        return model


CLASSES = 10
ROWS, COLS, CHS = 28, 28, 1

TRAIN_SIZE = 50000
TEST_SIZE = 10000
VALIDATION_SPLIT = TEST_SIZE / TRAIN_SIZE

BATCH_SIZE = 128
EPOCHS = 16

if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    model = LeNet().build(input_shape=(ROWS, COLS, CHS), classes=CLASSES)

    # keras.utils.plot_model(mode, to_file='model.png')

    model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), loss=keras.losses.categorical_crossentropy, metrics=[keras.metrics.categorical_accuracy])

    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, Y_test), verbose=2)

    score = model.evaluate(X_test, Y_test, verbose=0)

    print("loss: ", score[0])
    print("acc: ", score[1])

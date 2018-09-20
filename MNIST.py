# -*- coding: utf-8 -*-

from tensorflow import keras

from layers import dense, relu, softmax
from datasets import mnist

BATCH_SIZE = 128
EPOCHS = 8


def build(input_shape):
    model = keras.Sequential([
        dense(100, input_shape=input_shape),
        relu(),
        dense(300),
        relu(),
        dense(10),
        softmax()
    ])

    model.summary()

    return model


if __name__ == '__main__':

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data(normalize=False, flatten=True, one_hot_label=True)

    model = build((784, ))

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test)

    print("loss: ", score[0])
    print("acc: ", score[1])
# -*- coding: utf-8 -*-

from matplotlib import pyplot
from tensorflow import keras

CLASSES = 10
ROWS, COLS, CHS = 28, 28, 1

TRAIN_SIZE = 60000
TEST_SIZE = 10000

BATCH_SIZE = 128
EPOCHS = 32


def dense(units, **kwargs):
    return keras.layers.Dense(units,
                              kernel_regularizer=keras.regularizers.l2(0.0001),
                              bias_regularizer=keras.regularizers.l2(0.0001),
                              **kwargs)


def flatten(**kwargs):
    return keras.layers.Flatten(**kwargs)


def sigmoid(**kwargs):
    return keras.layers.Activation(activation=keras.activations.sigmoid, **kwargs)


def softmax(axis=-1, **kwargs):
    return keras.layers.Activation(activation=keras.activations.softmax, **kwargs)


class LeNet:
    """
    LeNet-300-100
    """

    def build(self, input_shape, classes):
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


def prepare():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    X_train /= 255
    X_test /= 255

    if keras.backend.image_data_format() == 'channels_last':
        X_train = X_train.reshape((TRAIN_SIZE, ROWS, COLS, CHS))
        X_test = X_test.reshape((TEST_SIZE, ROWS, COLS, CHS))
    else:
        X_train = X_train.reshape((TRAIN_SIZE, CHS, ROWS, COLS))
        X_test = X_test.reshape((TEST_SIZE, CHS, ROWS, COLS))

    Y_train = keras.utils.to_categorical(Y_train, CLASSES)
    Y_test = keras.utils.to_categorical(Y_test, CLASSES)

    return (X_train, Y_train), (X_test, Y_test)


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


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = prepare()

    model = LeNet().build(input_shape=(ROWS, COLS, CHS), classes=CLASSES)

    keras.utils.plot_model(model, to_file='model.png')

    model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    history = model.fit(X_train, Y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=2,
                        validation_data=(X_test, Y_test))

    plot(history, metrics=['loss', 'acc'])

    score = model.evaluate(X_test, Y_test, verbose=0)

    model.save('model.h5')

    print("loss: ", score[0])
    print("acc: ", score[1])

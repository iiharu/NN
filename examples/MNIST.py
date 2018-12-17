# -*- coding: utf-8 -*-

from matplotlib import pyplot
from tensorflow import keras
from tensorflow.keras import backend as K

# TODO:
# - Test group normalization with LeNet5.


CLASSES = 10
ROWS, COLS, CHS = 28, 28, 1

TRAIN_SIZE = 60000
TEST_SIZE = 10000

BATCH_SIZE = 128
EPOCHS = 32


class GroupNormalization(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[1:],
                                     initializer='ones',
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)
        super(GroupNormalization, self).build(input_shape)

    def call(self, x):
        G = 2
        N, H, W, C = K.int_shape(x)

        # x = K.reshape(x, (K.shape(x)[0], G, H, W, C // G))
        x = K.reshape(x, (K.shape(x)[0], C, H, W))
        x = K.reshape(x, (K.shape(x)[0], G, C // G, H, W))

        mean = K.mean(x, axis=0)
        var = K.var(x, axis=0)
        y = x - mean
        y = y / (K.sqrt(var) + K.epsilon())
        # y = K.reshape(y, (K.shape(x)[0], H, W, C))
        y = K.reshape(y, (K.shape(x)[0], C, H, W))
        y = K.reshape(y, (K.shape(x)[0], H, W, C))

        y = self.gamma * y + self.beta
        return y

    def compute_output_shape(self, input_shape):
        return input_shape


def activation(activation, **kwargs):
    return keras.layers.Activation(activation=activation, **kwargs)


def batch_normalization(axis=-1, **kwargs):
    return keras.layers.BatchNormalization(axis=axis, **kwargs)


def conv2d(filters, kernel_size, strides=1, padding='same',
           activation=None, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None,
           **kwargs):
    return keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                               activation=activation, use_bias=use_bias,
                               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                               kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                               kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                               **kwargs)


def dense(units, **kwargs):
    return keras.layers.Dense(units,
                              kernel_regularizer=keras.regularizers.l2(0.0001),
                              bias_regularizer=keras.regularizers.l2(0.0001),
                              **kwargs)


def flatten(**kwargs):
    return keras.layers.Flatten(**kwargs)


def max_pooling2d(pool_size=(2, 2), strides=None, padding='same', **kwargs):
    return keras.layers.MaxPooling2D(pool_size=pool_size,
                                     strides=strides,
                                     padding=padding,
                                     **kwargs)


def relu(max_value=None, **kwargs):
    return keras.layers.Activation(activation=activations.relu, **kwargs)


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


class LeNet5:

    """
    LeNet-5

    Modify:
    - This model uses relu as activation function but original one uses sigmoid.

    Example:
    model = LeNet5().build()
    """

    def __init__(self, activation='relu'):
        self.activation = activation

    def build(self, input_shape=(28, 28, 1), classes=10):
        inputs = keras.Input(shape=input_shape)

        outputs = conv2d(filters=6, kernel_size=(6, 6))(inputs)
        outputs = max_pooling2d(pool_size=(2, 2), strides=(2, 2))(outputs)
        outputs = activation(self.activation)(outputs)

        outputs = GroupNormalization()(outputs)
        outputs = conv2d(filters=16, kernel_size=(6, 6))(outputs)
        outputs = max_pooling2d(pool_size=(2, 2), strides=(2, 2))(outputs)
        outputs = activation(self.activation)(outputs)

        outputs = flatten()(outputs)
        outputs = dense(120)(outputs)
        outputs = activation(self.activation)(outputs)

        outputs = dense(64)(outputs)
        outputs = activation(self.activation)(outputs)

        outputs = dense(classes)(outputs)
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


def build(input_shape, classes):
    # model = LeNet().build(input_shape=input_shape, classes=classes)
    model = LeNet5().build(input_shape=input_shape, classes=classes)

    return model


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

    model = build(input_shape=(ROWS, COLS, CHS), classes=CLASSES)

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

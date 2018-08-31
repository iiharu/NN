# -*- coding: utf-8 -*-

from tensorflow import keras

from utils import plot

CLASSES = 10
ROWS, COLS, CHS = 32, 32, 3

TRAIN_SIZE = 50000
TEST_SIZE = 10000
VALIDATION_SPLIT = TEST_SIZE / TRAIN_SIZE

BATCH_SIZE = 128
EPOCHS = 32


def prepare():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
    Y_train = keras.utils.to_categorical(Y_train, CLASSES)
    Y_test = keras.utils.to_categorical(Y_test, CLASSES)

    return (X_train, Y_train), (X_test, Y_test)


def add():
    return keras.layers.Add()


def batch_normalization():
    return keras.layers.BatchNormalization()


def conv2d(filters, kernel_size, strides=1):
    return keras.layers.Conv2D(filters,
                               kernel_size,
                               strides=strides,
                               padding='same',
                               kernel_initializer=keras.initializers.he_normal(),
                               kernel_regularizer=keras.regularizers.l2(0.0001),
                               use_bias=False)


def dense(units):
    return keras.layers.Dense(units,
                              kernel_regularizer=keras.regularizers.l2(0.0001))


def global_average_pooling2d():
    return keras.layers.GlobalAveragePooling2D()


def relu():
    # return keras.layers.ReLU()
    return keras.layers.Activation(keras.activations.relu)


def softmax():
    # return keras.layers.Softmax()
    return keras.layers.Activation(keras.activations.softmax)


def residual(inputs, filters, kernel_size, down_sampling=False):
    """
    Residual Block (option B.)

    # Examples
    outputs = residual(outputs, 16, (3, 3), down_sampling=True)

    # Arguments
    inputs: inputs for residual block
    filters: filters for convolution layer
    kernel_size: kernel size for convolution layrt
    down_sampling: if true filters are doubled and image col and row are halved.
        (col, row, filter): (32, 32, 16) => (16, 16, 32)

    # References
        - [Deep Residual Learning for Image Recognition] (http://arxiv.org/abs/1512.03385)
    """

    if down_sampling:
        outputs = conv2d(filters, kernel_size, strides=2)(inputs)
    else:
        outputs = conv2d(filters, kernel_size)(inputs)

    outputs = batch_normalization()(outputs)
    outputs = relu()(outputs)
    outputs = conv2d(filters, kernel_size)(outputs)

    if down_sampling:
        inputs = conv2d(filters, kernel_size, strides=2)(inputs)

    outputs = add()([inputs, outputs])
    outputs = relu()(outputs)

    return outputs


def build(input_shape, n=3):
    """

    | output map size | 32x32 | 16 x 16 | 8x8 |
    | # layers        | 1+2n  | 2n      | 2n  |
    | # filters       | 16    | 32      | 64  |
    :param input_shape:
    :param n: Variant for ResNet Cifar-10
            3 => 20 layer
            5 => 32 layer
            7 => 44 layer
            9 => 56 layer
            18 => 110 layer
    :return:
    """
    inputs = keras.Input(shape=input_shape)

    outputs = batch_normalization()(inputs)
    outputs = conv2d(16, (3, 3))(outputs)

    for i in range(n):
        outputs = residual(outputs, 16, (3, 3))

    for i in range(n):
        if i == 0:
            outputs = residual(outputs, 32, (3, 3), down_sampling='same')
        else:
            outputs = residual(outputs, 32, (3, 3))

    for i in range(n):
        if i == 0:
            outputs = residual(outputs, 64, (3, 3), down_sampling='same')
        else:
            outputs = residual(outputs, 64, (3, 3))

    outputs = global_average_pooling2d()(outputs)
    outputs = dense(CLASSES)(outputs)
    outputs = softmax()(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    # To plot model, commnet out below line.
    # keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    return model


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = prepare()

    model = build(input_shape=(ROWS, COLS, CHS,), n=3)

    model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    history = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=2,
                        # callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, mode='auto')],
                        validation_split=VALIDATION_SPLIT)

    plot(history, metrics=['loss', 'acc'])

    score = model.evaluate(X_test, Y_test, verbose=0)

    print("loss: ", score[0])
    print("acc:  ", score[1])

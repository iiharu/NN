# -*- coding: utf-8 -*-

from tensorflow import keras

from layers import concat, batch_normalization, relu, average_pooling2d, global_average_pooling2d, softmax, dense

CLASSES = 10

GROWTH_RATE = 12

def prepare():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
    Y_train = keras.utils.to_categorical(Y_train, CLASSES)
    Y_test = keras.utils.to_categorical(Y_test, CLASSES)

    return (X_train, Y_train), (X_test, Y_test)


def conv2d(filters, kernel_size, strides=1):
    return keras.layers.Conv2D(filters,
                               kernel_size,
                               strides=strides,
                               padding='same',
                               kernel_initializer=keras.initializers.he_normal(),
                               kernel_regularizer=keras.regularizers.l2(0.0001),
                               use_bias=False)


def dense_conv(inputs, growth_rate, layers, kernel_size=(3, 3), down_sampling=False):

    outputs = []

    for k in range(layers):
        if k <= 1:
            inputs = batch_normalization()(inputs)
            inputs = relu()(inputs)
            inputs = conv2d(growth_rate, kernel_size)(inputs)
        else:
            inputs = concat()(outputs)
            inputs = batch_normalization()(inputs)
            inputs = relu()(inputs)
            inputs = conv2d(growth_rate, kernel_size)(inputs)
        outputs.append(inputs)

    outputs = concat()(outputs)

    return outputs


def transition(inputs, filters=12):
    outputs = conv2d(filters, kernel_size=(1, 1))(inputs)
    outputs = average_pooling2d(pool_size=(2, 2), strides=2)(outputs)
    return outputs

def build(input_shape):

    k = GROWTH_RATE
    l = 4 # num of conv layer in block
    n = 3 # num of dense block

    inputs = keras.Input(shape=input_shape)

    outputs = conv2d(16, kernel_size=(3, 3))(inputs)


    for i in range(n):
        outputs = dense_conv(outputs, growth_rate=k, layers=l)
        outputs = transition(outputs, (i + 1) * k * l)

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

    model = build((32, 32, 3))

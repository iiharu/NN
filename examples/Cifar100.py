# -*- coding: utf-8 -*-

from matplotlib import pyplot
from tensorflow import keras

CLASSES = 100
ROWS, COLS, CHS = 32, 32, 3

TRAIN_SIZE = 50000
TEST_SIZE = 10000

BATCH_SIZE = 128
EPOCHS = 32


def add(**kwargs):
    return keras.layers.Add(**kwargs)


def average_pooling2d(pool_size=(2, 2), strides=None, padding='same', **kwargs):
    return keras.layers.AveragePooling2D(pool_size=pool_size,
                                         strides=strides,
                                         padding=padding,
                                         **kwargs)


def batch_normalization(axis=-1, **kwargs):
    return keras.layers.BatchNormalization(axis=axis, **kwargs)


def conv2d(filters, kernel_size, strides=1, **kwargs):
    return keras.layers.Conv2D(filters,
                               kernel_size,
                               strides=strides,
                               padding='same',
                               use_bias=False,
                               kernel_initializer=keras.initializers.he_normal(),
                               kernel_regularizer=keras.regularizers.l2(
                                   0.0001),
                               **kwargs)


def dense(units, **kwargs):
    return keras.layers.Dense(units,
                              kernel_regularizer=keras.regularizers.l2(0.0001),
                              bias_regularizer=keras.regularizers.l2(0.0001),
                              **kwargs)


def flatten(**kwargs):
    return keras.layers.Flatten(**kwargs)


def global_average_pooling2d(**kwargs):
    return keras.layers.GlobalAveragePooling2D(**kwargs)


def relu(max_value=None, **kwargs):
    return keras.layers.Activation(activation=keras.activations.relu, **kwargs)


def softmax(axis=-1, **kwargs):
    # return keras.layers.Softmax(axis=axis, **kwargs)
    return keras.layers.Activation(activation=keras.activations.softmax, **kwargs)


class ResNet:

    """
    Residual Network

    Parameter:
    - blocks: num of blocks as list
    - filters: num of filter map
    - bottleneck: If True use bottleneck architecture
    - input_layers: input layers for network
        for ImageNet: input_layers=[conv2d(filters=64, kernel_size=(7, 7), strides=2)] (default)
        for Cifar: input_layers=[conv2d(filters=16, kernel_size=(3, 3))]
    - output_layers: output layers for network
        for ImageNet: output_layers=[average_pooling2d(pool_size=(2, 2)), flatten()])
        for Cifar: output_layers=[global_average_pooling2d()]

    References
    - [Deep Residual Learning for Image Recognition] (http://arxiv.org/abs/1512.03385)
    - [Identity Mappings in Deep Residual Networks] (https://arxiv.org/abs/1603.05027)
    """

    def __init__(self, blocks, filters, bottleneck=False,
                 input_layers=[
                     batch_normalization(),
                     relu(),
                     conv2d(filters=64, kernel_size=(7, 7), strides=2),
                 ],
                 output_layers=[average_pooling2d(pool_size=(2, 2)), flatten()]):
        self.blocks = blocks
        self.filters = filters
        self.bottleneck = bottleneck
        self.bn_axis = -1 if keras.backend.image_data_format() == 'channels_last' else 1
        self.input_layers = input_layers
        self.output_layer = output_layers

    def residual(self, inputs, filters, bottleneck=False, sub_sampling=False):
        strides = 2 if sub_sampling else 1
        kernel_size = (1, 1) if bottleneck else (3, 3)

        outputs = batch_normalization()(inputs)
        outputs = relu()(outputs)
        outputs = conv2d(filters=filters, kernel_size=kernel_size,
                         strides=strides)(outputs)
        if sub_sampling or bottleneck:
            # inputs = batch_normalization()(inputs)
            # inputs = relu()(inputs)
            inputs = conv2d(filters=4 * filters if bottleneck else filters,
                            kernel_size=(1, 1),
                            strides=strides)(inputs)
        outputs = batch_normalization()(outputs)
        outputs = relu()(outputs)
        outputs = conv2d(filters=filters, kernel_size=(
            3, 3), strides=1)(outputs)
        if bottleneck:
            outputs = batch_normalization()(outputs)
            outputs = relu()(outputs)
            outputs = conv2d(filters=4*filters,
                             kernel_size=kernel_size, strides=1)(outputs)
        outputs = add()([inputs, outputs])

        return outputs

    def build(self, input_shape, classes=1000):
        inputs = keras.Input(shape=input_shape)

        outputs = inputs
        for layer in self.input_layers:
            outputs = layer(outputs)

        for b in range(len(self.blocks)):
            for l in range(self.blocks[b]):
                sub_sampling = True if b != 0 and l == 0 else False
                outputs = self.residual(outputs, self.filters[b],
                                        bottleneck=self.bottleneck, sub_sampling=sub_sampling)

        for layer in self.output_layer:
            outputs = layer(outputs)

        outputs = dense(classes)(outputs)
        outputs = softmax()(outputs)

        model = keras.Model(inputs, outputs)

        model.summary()

        return model


def ResNet20():
    """
    ResNet for Cifar10/100 with 20 layers.
    """
    return ResNet(blocks=[3, 3, 3], filters=[16, 32, 64], input_layers=[conv2d(filters=16, kernel_size=(3, 3))], output_layers=[global_average_pooling2d()])


def ResNet32():
    """
    ResNet for Cifar10/100 with 32 layers.
    """
    return ResNet(blocks=[5, 5, 5], filters=[16, 32, 64], input_layers=[conv2d(filters=16, kernel_size=(3, 3))], output_layers=[global_average_pooling2d()])


def ResNet44():
    """
    ResNet for Cifar10/100 with 44 layers.
    """
    return ResNet(blocks=[7, 7, 7], filters=[16, 32, 64], input_layers=[conv2d(filters=16, kernel_size=(3, 3))], output_layers=[global_average_pooling2d()])


def ResNet56():
    """
    ResNet for Cifar10/100 with 56 layers.
    """
    return ResNet(blocks=[9, 9, 9], filters=[16, 32, 64], input_layers=[conv2d(filters=16, kernel_size=(3, 3))], output_layers=[global_average_pooling2d()])


def prepare():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar100.load_data(
        label_mode='fine')

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    X_train /= 255
    X_test /= 255

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

    model = ResNet56().build(input_shape=(ROWS, COLS, CHS), classes=CLASSES)

    keras.utils.plot_model(model, to_file='model.png')

    model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy, keras.metrics.top_k_categorical_accuracy])

    datagen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=4,
                                                           height_shift_range=4,
                                                           fill_mode='constant',
                                                           horizontal_flip=True)

    datagen.fit(X_train)

    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                  steps_per_epoch=TRAIN_SIZE // 1,
                                  epochs=EPOCHS,
                                  verbose=2,
                                  validation_data=(X_test, Y_test))

    plot(history, metrics=[
         'loss', 'categorical_accuracy', 'top_k_categorical_accuracy'])

    score = model.evaluate(X_test, Y_test, verbose=0)

    print("loss: ", score[0])
    print("acc:  ", score[1])

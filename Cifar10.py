# -*- coding: utf-8 -*-

import numpy as np
from tensorflow import keras

from models import ResNet
from models import ResNet20
from models import ResNet32
from models import ResNet44
from models import ResNet56

from utils import plot
from datasets import cifar10

CLASSES = 10
ROWS, COLS, CHS = 32, 32, 3

TRAIN_SIZE = 50000
TEST_SIZE = 10000
VALIDATION_SPLIT = TEST_SIZE / TRAIN_SIZE
VALIDATION_SIZE = 5000

BATCH_SIZE = 128
EPOCHS = 32


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    model = ResNet44().build(input_shape=(ROWS, COLS, CHS, ), classes=CLASSES)

    model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    # We follow the simple data augmentation in "Deeply-supervised nets" (http://arxiv.org/abs/1409.5185) for training:
    # 4 pixels are padded on each side,
    # and a 32×32 crop is randomly sampled from the padded image or its horizontal flip.
    # For testing, we only evaluate the single view of the original 32×32 image.
    datagen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=4,
                                                           height_shift_range=4,
                                                           fill_mode='constant',
                                                           horizontal_flip=True)

    datagen.fit(X_train)

    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                  steps_per_epoch=TRAIN_SIZE // 10,
                                  epochs=EPOCHS,
                                  verbose=2,
                                  # callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, mode='auto')],
                                  validation_data=(X_test, Y_test))

    plot(history, metrics=['loss', 'acc'])

    score = model.evaluate(X_test, Y_test, verbose=0)

    print("loss: ", score[0])
    print("acc:  ", score[1])

# -*- coding: utf-8 -*-

from tensorflow import keras

MAX_FEATURES = 20000
MAXLEN = 80
BATCH_SIZE = 32


def prepare(num_words=None, maxlen=None):
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.imdb.load_data(num_words=num_words,
                                                                         skip_top=0,
                                                                         maxlen=maxlen,
                                                                         seed=113,
                                                                         start_char=1,
                                                                         oov_char=2,
                                                                         index_from=3)

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print('Pad sequences (samples x time)')
    X_train = keras.preprocessing.sequence.pad_sequences(
        X_train, maxlen=maxlen)
    X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    return (X_train, Y_train), (X_test, Y_test)


def build():

    model = keras.Sequential([
        keras.layers.Embedding(MAX_FEATURES, 128),
        keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()

    return model


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = prepare(
        num_words=MAX_FEATURES, maxlen=MAXLEN)

    model = build()

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['acc'])

    model.fit(X_train, Y_train,
              batch_size=BATCH_SIZE,
              epochs=15,
              verbose=2,
              validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=2)

    print('loss:', score[0])
    print('acc:', score[1])

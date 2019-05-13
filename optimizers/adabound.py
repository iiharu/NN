
import tensorflow as tf
from tensorflow import keras
# from tensorflow._api.v1.keras.optimizers import K, Optimizer, math_ops, state_ops
# from tensorflow.keras.optimizers import K, Optimizer, math_ops, state_ops
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops, state_ops
from tensorflow.keras.optimizers import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent optimizer.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, **kwargs):
        super(SGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [state_ops.assign_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * \
                (1. / (1.+self.decay * math_ops.cast(self.iterations, K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(state_ops.assign(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'momentum': float(K.get_value(self.momentum)),
            'decay': float(K.get_value(self.decay)),
            'nesterov': self.nesterov
        }
        base_config = super(SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    Y_train = keras.utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.to_categorical(Y_test, 10)
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(784),
        keras.layers.Activation(keras.activations.relu),
        keras.layers.Dense(10),
        keras.layers.Activation(keras.activations.softmax),
    ])
    model.compile(optimizer=SGD(lr=0.01), loss=keras.losses.categorical_crossentropy, metrics=[
                  keras.metrics.categorical_accuracy])
    model.fit(X_train, Y_train, batch_size=16, epochs=16,
              verbose=2, validation_data=(X_test, Y_test))

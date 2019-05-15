
# import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.optimizers import K, Optimizer, math_ops, state_ops
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, state_ops


class SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    !!!Use CPU only!!!
    """

    def __init__(self,
                 lr=0.01,
                 momentum=0.,
                 decay=0.,
                 nesterov=False,
                 **kwargs):
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
            lr = lr * (1. / (1. + self.decay *
                             math_ops.cast(self.iterations,
                                           K.dtype(self.decay))))
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


class Adam(Optimizer):
    """Adam optimizer.

    !!!Use CPU only!!!
    """

    def __init__(self,
                 lr=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=None,
                 decay=0.,
                 amsgrad=False,
                 **kwargs):
        super(Adam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (
                1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                      K.dtype(self.decay))))

        with ops.control_dependencies([state_ops.assign_add(self.iterations,
                                                            1)]):
            t = math_ops.cast(self.iterations, K.floatx())
        lr_t = lr * (K.sqrt(1. - math_ops.pow(self.beta_2, t))
                     / (1. - math_ops.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g)
            if self.amsgrad:
                vhat_t = math_ops.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(state_ops.assign(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad
        }
        base_config = super(Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdaBound(Optimizer):
    """AdaBound optimizer.

    Default parameters follow those provided in the original paper.

    Arguments:
        alpha: float >= 0
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        # decay: float >= 0. Learning rate decay over each update.
        # amsbound:
    """

    def __init__(self,
                 alpha=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=None,
                 # decay=0.,
                 amsbound=False,
                 **kwargs):
        super(AdaBound, self).__init(**kwargs)
        with K.name_scope(self.__name__.__class__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.alpha = K.variable(alpha, name='alpha')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            # self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        # self.initial_decay = decay
        self.amsbound = amsbound

    def eta_l(self, t):
        return 0.1 - 0.1 / ((1. - self.beta_2) * t + 1. + self.epsilon)

    def eta_u(self, t):
        return 0.1 + 0.1 / ((1. - self.beta_2) * t + self.epsilon)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = []

        alpha = self.alpha
        # if self.initial_decay > 0:
        #     alpha = alpha * \
        #         (1. / (1. + self.decay
        #                * math_ops.cast(self.iterations, K.dtype(self.decay))))

        with ops.control_dependencies([state_ops.assign_add(self.iterations, 1)]):
            t = math_ops.cast(self.iterations, K.floatx())
        # alpha_t = alpha * (K.sqrt(1. - math_ops.pow(self.beta_2, t)
        #                           ) / (1. - math_ops.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsbound:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, q, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g)
            etahat_t = K.clip(alpha / math_ops.sqrt(v_t), eta_l(t), eta_u(t))
            eta_t = etahat_t / math_ops.sqrt(t)
            if self.amsbound:
                vhat_t = math_ops.maximum(vhat, v_t)
                p_t = p - eta_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(state_ops.assign(vhat, vhat_t))
            else:
                p_t = p - alpha_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', Node) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'alpha': float(K.get_value(self.alpha)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            # 'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon
            # 'amsbound': self.amsbound
        }
        base_config = super(AdaBound, self).get_config()
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
    model.compile(optimizer=SGD(lr=0.01),
                  loss=keras.losses.categorical_crossentropy, metrics=[
                  keras.metrics.categorical_accuracy])
    model.fit(X_train, Y_train, batch_size=16, epochs=16,
              verbose=2, validation_data=(X_test, Y_test))

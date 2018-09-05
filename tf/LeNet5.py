# -*- coding: utf-8 -*-

import tensorflow as tf

# from datasets import mnist

from tensorflow.examples.tutorials.mnist import input_data


def inference(x, n_out=None, activation=None):

    conv1 = tf.layers.conv2d(x, filters=32,
                             kernel_size=[5, 5],
                             padding='same',
                             activation=activation)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(pool1, filters=64,
                             kernel_size=[5, 5],
                             padding='same',
                             activation=activation)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)
    flattend = tf.layers.flatten(pool2)
    dense = tf.layers.dense(flattend, units=1024, activation=activation)

    # Logits Layer
    y = tf.layers.dense(dense, units=n_out)

    return y


def loss(y, t):
    with tf.name_scope('loss'):
        cross_entropy = tf.losses.softmax_cross_entropy(t, y)
    return cross_entropy


def training(loss):
    optimizer = tf.train.MomentumOptimizer(0.01, 0.9, use_nesterov=True)
    with tf.name_scope('train'):
        train_step = optimizer.minimize(loss)
    return train_step


def accuracy(y, t):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argxmax(y, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

if __name__ == '__main__':

    # Prepare dataset

    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    t = tf.placeholder(tf.float32, shape=(None, 10))
    y= inference(x, n_out=10, activation=tf.nn.relu)
    loss = loss(y, t)
    train_step = training(loss)
    accuracy = accuracy(y, t)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

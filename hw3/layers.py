#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Input


class Embedding(tf.keras.Model):
    def __init__(self, num_words, embeddings, dim):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            num_words + 1, dim, weights=[embeddings], trainable=False, mask_zero=True
        )

    def call(self, x):
        return self.embedding(x)


class BiLSTM(tf.keras.Model):
    def __init__(self, units):
        super().__init__()

        self.bilstm = Bidirectional(
            LSTM(
                units,
                return_sequences=True,
                recurrent_initializer="glorot_uniform",
                dropout=0.3,
            )
        )

    def call(self, x):
        return self.bilstm(x)


class BiLinear_Layer(tf.keras.Model):
    def __init__(self, units, length):
        super().__init__()

        weight_init = tf.random_normal_initializer()
        self.W1S = tf.Variable(
            initial_value=weight_init((units, units)), dtype="float32", trainable=True
        )
        self.W2S = tf.Variable(
            initial_value=weight_init((length, 1)), dtype="float32", trainable=True
        )

        self.W1E = tf.Variable(
            initial_value=weight_init((units, units)), dtype="float32", trainable=True
        )
        self.W2E = tf.Variable(
            initial_value=weight_init((length, 1)), dtype="float32", trainable=True
        )

    def call(self, context_mat, question_mat):
        tmp1 = context_mat @ self.W1S
        tmp1 = tmp1 @ tf.transpose(question_mat, [0, 2, 1])
        tmp1 = tmp1 @ self.W2S
        tmp1 = tf.nn.softmax(tmp1, axis=1)

        tmp2 = context_mat @ self.W1E
        tmp2 = tmp2 @ tf.transpose(question_mat, [0, 2, 1])
        tmp2 = tmp2 @ self.W2E
        tmp2 = tf.nn.softmax(tmp2, axis=1)

        return tf.concat([tmp1, tmp2], axis=1)

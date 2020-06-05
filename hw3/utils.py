#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

con_len = None


class Custom_Accuracy(tf.keras.metrics.Metric):
    def __init__(self, nm, name="custom_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.nm = nm
        self.accuracy = self.add_weight(name="tp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        len_ = np.shape(y_pred)[1]

        y_1 = y_pred[:, : len_ // 2]
        y_2 = y_pred[:, len_ // 2 :]
        y_1 = tf.reshape(tf.argmax(y_1, axis=1), shape=(-1, 1))
        y_2 = tf.reshape(tf.argmax(y_2, axis=1), shape=(-1, 1))

        y_1t = y_true[:, : len_ // 2]
        y_2t = y_true[:, len_ // 2 :]

        y_1t = tf.reshape(tf.argmax(y_1t, axis=1), shape=(-1, 1))
        y_2t = tf.reshape(tf.argmax(y_2t, axis=1), shape=(-1, 1))

        values1 = tf.cast(y_1, "int32") == tf.cast(y_1t, "int32")
        values1 = tf.cast(values1, "float32")

        values2 = tf.cast(y_2, "int32") == tf.cast(y_2t, "int32")
        values2 = tf.cast(values2, "float32")

        self.accuracy.assign_add(
            (tf.reduce_sum(values1 + values2))
            / (tf.dtypes.cast(2 * self.nm, tf.float32))
        )

    def result(self):
        return self.accuracy


def loss(y_true, prob):
    start_label = y_true[:, :con_len]
    end_label = y_true[:, con_len:]

    start_logit = prob[:, :con_len]
    end_logit = prob[:, con_len:]

    start_loss = tf.keras.backend.categorical_crossentropy(start_label, start_logit)
    end_loss = tf.keras.backend.categorical_crossentropy(end_label, end_logit)

    return start_loss + end_loss

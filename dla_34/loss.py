# -*- coding: utf-8 -*-
"""
@author: wi-ith
"""
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS


def soft_max(logits, axis=-1):
    tile_depth = logits.shape[axis]
    max_value = tf.tile(tf.reshape((tf.reduce_max(logits, axis=axis)), [-1, 1]), [1, tile_depth])
    exp_logits = tf.exp(logits-max_value)
    exp_sum = tf.tile(tf.reshape((tf.reduce_sum(exp_logits, axis=axis)), [-1, 1]), [1, tile_depth])

    return exp_logits / exp_sum

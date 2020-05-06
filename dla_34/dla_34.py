# -*- coding: utf-8 -*-
"""
@author: wi-ith
"""


import tensorflow as tf

import model

import tensorflow.contrib.slim as slim


FLAGS = tf.app.flags.FLAGS

def soft_max(logits, axis=-1):
    tile_depth = logits.shape[axis]
    max_value = tf.tile(tf.reshape((tf.reduce_max(logits, axis=axis)), [-1, 1]), [1, tile_depth])
    exp_logits = tf.exp(logits-max_value)
    exp_sum = tf.tile(tf.reshape((tf.reduce_sum(exp_logits, axis=axis)), [-1, 1]), [1, tile_depth])

    return exp_logits / exp_sum

def inference(images):

    with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):

        inference_model = model.DLA_34(is_training=False, input_size=FLAGS.image_size)

        logits=inference_model._build_model(images)

        prediction = soft_max(logits,-1)

        return prediction

def loss(images, labels):

    with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):

        train_model = model.DLA_34(is_training=True, input_size=FLAGS.image_size)

        logits=train_model._build_model(images)

        loss = tf.losses.sparse_softmax_cross_entropy(labels,logits)

        # loss = tf.losses.softmax_cross_entropy(logits,labels)

    return loss




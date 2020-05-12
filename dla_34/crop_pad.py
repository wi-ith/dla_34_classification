# -*- coding: utf-8 -*-
"""
@author: wi-ith
"""
import tensorflow as tf
from tensorflow.python.ops import math_ops

FLAGS = tf.app.flags.FLAGS



def random_crop_image(image,
                      labels,
                      min_object_covered=0.05,
                      aspect_ratio_range=(0.75, 1.33),
                      area_range=(0.08, 1.0),
                      overlap_thresh=0.3,
                      clip_boxes=True):

    image_shape = tf.shape(image)
    bb_box=tf.reshape(tf.convert_to_tensor([0., 0., 1., 1.]),[1,1,-1])
    im_box_begin, im_box_size, im_box = tf.image.sample_distorted_bounding_box(
        image_shape,
        bounding_boxes=bb_box,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    new_image = tf.slice(image, im_box_begin, im_box_size)
    new_image.set_shape([None, None, image.get_shape()[2]])

    return new_image, labels


def _random_integer(minval, maxval, seed):
    return tf.random_uniform(
        [], minval=minval, maxval=maxval, dtype=tf.int32, seed=seed)


def random_pad(image,
               min_image_size=None,
               max_image_size=None,
               pad_color=None,
               seed=None):
    if pad_color is None:
        pad_color = tf.reduce_mean(image, axis=[0, 1])

    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    max_image_size = tf.stack([image_height * 3, image_width * 3])
    max_image_size = tf.maximum(max_image_size,
                                tf.stack([image_height, image_width]))

    if min_image_size is None:
        min_image_size = tf.stack([image_height, image_width])
    min_image_size = tf.maximum(min_image_size,
                                tf.stack([image_height, image_width]))

    target_height = tf.cond(
        max_image_size[0] > min_image_size[0],
        lambda: _random_integer(min_image_size[0], max_image_size[0], seed),
        lambda: max_image_size[0])

    target_width = tf.cond(
        max_image_size[1] > min_image_size[1],
        lambda: _random_integer(min_image_size[1], max_image_size[1], seed),
        lambda: max_image_size[1])

    offset_height = tf.cond(
        target_height > image_height,
        lambda: _random_integer(0, target_height - image_height, seed),
        lambda: tf.constant(0, dtype=tf.int32))

    offset_width = tf.cond(
        target_width > image_width,
        lambda: _random_integer(0, target_width - image_width, seed),
        lambda: tf.constant(0, dtype=tf.int32))

    new_image = tf.image.pad_to_bounding_box(
        image,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=target_height,
        target_width=target_width)

    image_ones = tf.ones_like(image)
    image_ones_padded = tf.image.pad_to_bounding_box(
        image_ones,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=target_height,
        target_width=target_width)
    image_color_padded = (1.0 - image_ones_padded) * pad_color
    new_image += image_color_padded

    return new_image

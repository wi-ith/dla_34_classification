# -*- coding: utf-8 -*-
"""
@author: wi-ith
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.app.flags.DEFINE_string('ckpt_save_path', "./ckpt/imagenet/",'')

tf.app.flags.DEFINE_string('pretrained_ckpt_path', "",'')

tf.app.flags.DEFINE_string('mode', "train",'')

tf.app.flags.DEFINE_string('tfrecords_dir', "",'')

tf.app.flags.DEFINE_integer('image_size', 224,'')

tf.app.flags.DEFINE_integer('val_image_size', 256,'')

tf.app.flags.DEFINE_integer('num_train', 1200000,'')

tf.app.flags.DEFINE_integer('max_steps', 10000000,'')

tf.app.flags.DEFINE_integer('num_readers', 4,'')

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,'')

tf.app.flags.DEFINE_integer('input_queue_memory_factor', 4,'')

tf.app.flags.DEFINE_integer('num_classes', 1000,"")

tf.app.flags.DEFINE_integer('batch_size', 128,"")

tf.app.flags.DEFINE_integer('num_gpus', 1,'')

tf.app.flags.DEFINE_float('learning_rate', 0.01, '')

tf.app.flags.DEFINE_float('random_flip_prob', 0.5, '')

tf.app.flags.DEFINE_float('crop_prob', 0.15, '')

tf.app.flags.DEFINE_float('random_pad_prob', 0.0, '')

##validation

tf.app.flags.DEFINE_integer('num_validation', 50000,'')
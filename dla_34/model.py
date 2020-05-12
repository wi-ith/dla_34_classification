# -*- coding: utf-8 -*-
"""
@author: wi-ith
"""
import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.contrib.framework.python.ops import arg_scope

_WEIGHT_DECAY = 1e-4

FLAGS = tf.app.flags.FLAGS

class DLA_34(object):
    def __init__(self, is_training=True, input_size=224):
        self.input_size = input_size
        self.is_training = is_training
        self.normalizer = tc.layers.batch_norm
        self.bn_params = {'is_training': self.is_training,
                          'scale': True,
                          'center': True,
                          'decay': 0.9997,
                          'epsilon': 0.001,
                          }

    def basic_block(self, input, output_dims, stride=1, scope=None, dilation=1):
        with tf.variable_scope(scope+'_basic_block', reuse=tf.AUTO_REUSE):
            residual = tf.identity(input)
            input_dims = tf.shape(input)[0]

            if stride > 1:
                residual = tf.identity(tf.nn.max_pool(residual,
                                                      ksize=(1,2,2,1),
                                                      strides=(1,2,2,1),
                                                      padding='SAME'))

            if input_dims!=output_dims:
                residual = tc.layers.conv2d(residual, output_dims, 1,
                                      stride=1,
                                      activation_fn=None,
                                      normalizer_fn=self.normalizer,
                                      normalizer_params=self.bn_params,
                                      scope='project')



            output = tc.layers.conv2d(input, output_dims, 3,
                                      stride=stride,
                                      activation_fn=None,
                                      normalizer_fn=self.normalizer,
                                      normalizer_params=self.bn_params,
                                      scope='conv_1')
            output = tf.nn.relu6(output)
            output = tc.layers.conv2d(output, output_dims, 3,
                                      stride=1,
                                      activation_fn=None,
                                      normalizer_fn=self.normalizer,
                                      normalizer_params=self.bn_params,
                                      scope='conv_2')
            output = output + residual
            output = tf.nn.relu6(output)
            return output, residual


    def root_block(self, *input, scope=None, output_dims, residual=False):
        with tf.variable_scope(scope + '_root_block', reuse=tf.AUTO_REUSE):
            output = tf.concat(input, axis=3)
            output = tc.layers.conv2d(output, output_dims, 1,
                                      stride=1,
                                      activation_fn=None,
                                      normalizer_fn=self.normalizer,
                                      normalizer_params=self.bn_params,
                                      scope='conv_1')
            if residual:
                output = output + input[0]
            output = tf.nn.relu6(output)
            return output

    def _build_model(self, image):
        self.i = 0
        with arg_scope([tc.layers.conv2d],
                       weights_regularizer=tc.layers.l2_regularizer(_WEIGHT_DECAY)):
            with tf.variable_scope('FeatureExtractor/MobilenetV2', reuse=tf.AUTO_REUSE):
                # image_copy=tf.identity(image)
                #base_layer
                output = tc.layers.conv2d(image, 16, 7, 1,
                                          activation_fn=tf.nn.relu6,
                                          normalizer_fn=self.normalizer,
                                          normalizer_params=self.bn_params)  # base layer
                #level 0
                output = tc.layers.conv2d(output, 16, 3, 1,
                                          activation_fn=tf.nn.relu6,
                                          normalizer_fn=self.normalizer,
                                          normalizer_params=self.bn_params)  # level 0
                #level 1
                output = tc.layers.conv2d(output, 32, 3, 2,
                                          activation_fn=tf.nn.relu6,
                                          normalizer_fn=self.normalizer,
                                          normalizer_params=self.bn_params)  # level 1

                #level 2
                output_lv2_1, _ = self.basic_block(output, 64, stride=2, scope='output_lv2_1', dilation=1)
                output_lv2_2, _ = self.basic_block(output_lv2_1, 64, stride=1, scope='output_lv2_2', dilation=1)
                output_lv2_root = self.root_block(output_lv2_1,
                                                  output_lv2_2,
                                                  scope='output_lv2_root',
                                                  output_dims=64)

                #level 3
                output_lv3_1_1, output_lv3_1_1_residual = self.basic_block(output_lv2_root, 128, stride=2, scope='output_lv3_1_1', dilation=1)
                output_lv3_1_2, _ = self.basic_block(output_lv3_1_1, 128, stride=1, scope='output_lv3_1_2', dilation=1)
                output_lv3_1_root = self.root_block(output_lv3_1_1,
                                                    output_lv3_1_2,
                                                    scope='output_lv3_1_root',
                                                    output_dims=128)

                output_lv3_2_1, _ = self.basic_block(output_lv3_1_root, 128, stride=1, scope='output_lv3_1_2', dilation=1)
                output_lv3_2_2, _ = self.basic_block(output_lv3_2_1, 128, stride=1, scope='output_lv3_1_2', dilation=1)
                output_lv3_2_root = self.root_block(output_lv3_2_2,
                                                    output_lv3_2_1,
                                                    output_lv3_1_1_residual,
                                                    output_lv3_1_root,
                                                    scope='output_lv3_1_2',
                                                    output_dims=128)

                #level 4
                output_lv4_1_1, output_lv4_1_1_residual = self.basic_block(output_lv3_2_root, 256, stride=2, scope='output_lv4_1_1', dilation=1)
                output_lv4_1_2, _ = self.basic_block(output_lv4_1_1, 256, stride=1, scope='output_lv4_1_2', dilation=1)
                output_lv4_1_root = self.root_block(output_lv4_1_1,
                                                    output_lv4_1_2,
                                                    scope='output_lv4_1_root',
                                                    output_dims=256)

                output_lv4_2_1, _ = self.basic_block(output_lv4_1_root, 256, stride=1, scope='output_lv4_2_1', dilation=1)
                output_lv4_2_2, _ = self.basic_block(output_lv4_2_1, 256, stride=1, scope='output_lv4_2_2', dilation=1)
                output_lv4_2_root = self.root_block(output_lv4_2_2,
                                                    output_lv4_2_1,
                                                    output_lv4_1_1_residual,
                                                    output_lv4_1_root,
                                                    scope='output_lv4_2_root',
                                                    output_dims=256)

                #level 5
                output_lv5_1, output_lv5_1_residual = self.basic_block(output_lv4_2_root, 64, stride=2, scope='output_lv5_1', dilation=1)
                output_lv5_2, _ = self.basic_block(output_lv5_1, 64, stride=1, scope='output_lv5_2', dilation=1)
                output_lv5_root = self.root_block(output_lv5_1,
                                                  output_lv5_2,
                                                  output_lv5_1_residual,
                                                  scope='output_lv5_root',
                                                  output_dims=64)

                pool_size = FLAGS.image_size // 32
                avg_pool = tf.nn.avg_pool(output_lv5_root, (1,pool_size,pool_size,1), (1,1,1,1), padding='VALID')
                fc = tc.layers.conv2d(avg_pool, FLAGS.num_classes, 1, 1,
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      normalizer_params=None)
                fc = tf.reshape(fc,[-1,FLAGS.num_classes])
                return fc
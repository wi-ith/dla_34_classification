# -*- coding: utf-8 -*-
"""
@author: wi-ith
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from six.moves import xrange


import dla_34
import input
import flags


FLAGS = tf.app.flags.FLAGS


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


with tf.Graph().as_default(), tf.device('/cpu:0'):
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES],
        dtype=tf.float32)

    lr=FLAGS.learning_rate
    # opt = tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.9, epsilon=1)
    opt = tf.train.MomentumOptimizer(lr, momentum=0.9)

    # Get images and labels
    # for train
    with tf.name_scope('train_images'):
        images, labels = input.distorted_inputs(FLAGS.batch_size)

    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * FLAGS.num_gpus)

    tower_grads = []
    tower_losses = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ('tower', i)) as scope:

                    image_batch, label_batch = batch_queue.dequeue()

                    loss = dla_34.loss(image_batch, label_batch)

                    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

                    loss = loss + regularization_loss

                    tf.get_variable_scope().reuse_variables()

                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

                    grads = opt.compute_gradients(loss)

                    tower_grads.append(grads)
                    tower_losses.append(loss)

    total_loss = tf.reduce_mean(tower_losses)
    summaries.append(tf.summary.scalar('total_loss', total_loss))

    grads = average_gradients(tower_grads)

    #validation
    with tf.name_scope('eval_images'):
        val_images, val_labels = input.inputs(1)
    with tf.device('/gpu:0'):
        pred_val, labels_val = dla_34.inference(val_images,val_labels)

    summaries.extend(tf.get_collection(tf.GraphKeys.SUMMARIES, 'train_images'))
    summaries.extend(tf.get_collection(tf.GraphKeys.SUMMARIES, 'eval_images'))


    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        print(var.name)
        summaries.append(tf.summary.histogram(var.op.name, var))

    saver = tf.train.Saver(max_to_keep=20)

    summary_op = tf.summary.merge(summaries)

    pretrained_ckpt_path = FLAGS.pretrained_ckpt_path

    if pretrained_ckpt_path == "":
        print('no pretrained')
        init_fn = None

    elif  not tf.train.latest_checkpoint(FLAGS.ckpt_save_path):
        print('pretrained ckpt')
        exclude_layers = ['global_step']
        restore_variables = slim.get_variables_to_restore(exclude=exclude_layers)
        init_fn = slim.assign_from_checkpoint_fn(pretrained_ckpt_path,
                                                 restore_variables, ignore_missing_vars=True)

    else:
        print('training ckpt')
        init_fn = None


    sv = tf.train.Supervisor(logdir=FLAGS.ckpt_save_path,
                             summary_op=None,
                             saver=saver,
                             save_model_secs=0,
                             init_fn=init_fn)
    config_ = tf.ConfigProto(allow_soft_placement=True)
    config_.gpu_options.per_process_gpu_memory_fraction = 0.65

    # sess=sv.managed_session(config=config_)
    with sv.managed_session(config=config_) as sess:
        # Start the queue runners.
        sv.start_queue_runners(sess=sess)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            sess.run(train_op)
            sv_global_step, loss_value = sess.run([sv.global_step, loss])
            duration = time.time() - start_time


            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if sv_global_step % 100 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus
                Epoch_ = np.round(sv_global_step / (FLAGS.num_train / FLAGS.batch_size), 2)
                format_str = ('Epoch : %.2f   step %d, loss = %.2f (%.1f examples/sec; %.3f '
                            'sec/batch)')
                print (format_str % (Epoch_, sv_global_step, loss_value,
                                     examples_per_sec, sec_per_batch))

            if sv_global_step % 10 == 0:
                summary_str = sess.run(summary_op)
                sv.summary_computed(sess, summary_str)

            if sv_global_step % (int(FLAGS.num_train / FLAGS.batch_size)*4) == 0 and sv_global_step!=0:

                print('start validation')
                collect = 0
                for val_step in range(FLAGS.num_validation):

                    if val_step%5000==0:
                        print(val_step,' / ',FLAGS.num_validation)
                    val_cls_pred, val_GT = sess.run([pred_val,labels_val])
                    prediction_ = np.argmax(val_cls_pred)
                    if prediction_ == val_GT:
                        collect+=1
                accuracy_top1=collect/FLAGS.num_validation
                print(accuracy_top1," % ")

                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Top1', simple_value=float(accuracy_top1))
                sv.summary_computed(sess, summary)

            if sv_global_step % (int(FLAGS.num_train / FLAGS.batch_size) * 1) == 0 and sv_global_step != 0:
                checkpoint_path = os.path.join(FLAGS.ckpt_save_path, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=sv.global_step)



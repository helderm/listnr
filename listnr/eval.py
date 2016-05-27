# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for Classy"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os

import numpy as np
import tensorflow as tf

import timit as tm
import model as md

NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 5000
NUM_EXAMPLES_PER_EPOCH_FOR_DEV = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 106275

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../data/eval/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_type', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../data/train/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")


def _get_num_examples():
    eval_type = FLAGS.eval_type
    if eval_type == 'train':
        return NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    elif eval_type == 'test':
        return NUM_EXAMPLES_PER_EPOCH_FOR_TEST
    elif eval_type == 'dev':
        return NUM_EXAMPLES_PER_EPOCH_FOR_DEV
    else:
        raise Exception('Unknown eval_type')


def eval_once(num_examples, saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.

    Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)


        accuracies = []
        steps = []
        for path in ckpt.all_model_checkpoint_paths:

            # Restores from checkpoint
            ckpt_path = path if not os.path.isabs(FLAGS.checkpoint_dir) \
                            else os.path.join(FLAGS.checkpoint_dir, path)
            saver.restore(sess, ckpt_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt_path.split('/')[-1].split('-')[-1]
            steps.append(global_step)

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

                num_iter = int(math.ceil(num_examples / FLAGS.batch_size))
                true_count = 0  # Counts the number of correct predictions.
                total_sample_count = num_iter * FLAGS.batch_size
                step = 0

                while step < num_iter: #and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    correct_examples = np.sum(predictions)
                    true_count += correct_examples
                    step += 1

                # Compute precision @ 1.
                precision = true_count / (step * FLAGS.batch_size)
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
                accuracies.append(precision)

                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @ 1', simple_value=precision)
                summary_writer.add_summary(summary, global_step)
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads, stop_grace_period_secs=10)

        np.save(os.path.join(FLAGS.eval_dir, FLAGS.eval_type + '_accuracies'), np.array(accuracies))
        np.save(os.path.join(FLAGS.eval_dir, FLAGS.eval_type + '_steps'), np.array(steps))



def evaluate():
    """Eval Listnr for a number of steps."""

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    accuracies = []
    steps = []
    for path in ckpt.all_model_checkpoint_paths:

        with tf.Graph().as_default() as g:
            # Get images and labels for CIFAR-10.
            num_examples = _get_num_examples()
            images, labels = tm.inputs(FLAGS.batch_size, num_examples, type=FLAGS.eval_type, shuffle=False)

            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits = md.inference(images)

            # Calculate predictions.
            top_k_op = tf.nn.in_top_k(logits, labels, 1)

            # Restore the moving average version of the learned variables for eval.
            variable_averages = tf.train.ExponentialMovingAverage(
                md.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()

            summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

            with tf.Session() as sess:

                # Restores from checkpoint
                ckpt_path = path if not os.path.isabs(FLAGS.checkpoint_dir) \
                    else os.path.join(FLAGS.checkpoint_dir, path)
                saver.restore(sess, ckpt_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt_path.split('/')[-1].split('-')[-1]
                steps.append(global_step)

                # Start the queue runners.
                coord = tf.train.Coordinator()
                try:
                    threads = []
                    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                         start=True))

                    num_iter = int(math.ceil(num_examples / FLAGS.batch_size))
                    true_count = 0  # Counts the number of correct predictions.
                    total_sample_count = num_iter * FLAGS.batch_size
                    step = 0

                    while step < num_iter and not coord.should_stop():
                            predictions = sess.run([top_k_op])
                            correct_examples = np.sum(predictions)
                            true_count += correct_examples
                            step += 1

                    # Compute precision @ 1.
                    precision = true_count / (step * FLAGS.batch_size)
                    print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
                    accuracies.append(precision)

                    summary = tf.Summary()
                    summary.ParseFromString(sess.run(summary_op))
                    summary.value.add(tag='Precision @ 1', simple_value=precision)
                    summary_writer.add_summary(summary, global_step)
                except Exception as e:  # pylint: disable=broad-except
                    coord.request_stop(e)
                finally:
                    coord.request_stop()
                    coord.join(threads, stop_grace_period_secs=10)

    np.save(os.path.join(FLAGS.eval_dir, FLAGS.eval_type + '_accuracies'), np.array(accuracies))
    np.save(os.path.join(FLAGS.eval_dir, FLAGS.eval_type + '_steps'), np.array(steps))

                        #while True:
            #    eval_once(num_examples, saver, summary_writer, top_k_op, summary_op)
            #    if FLAGS.run_once:
            #        break
            #    time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()

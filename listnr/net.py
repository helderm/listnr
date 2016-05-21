#!/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import time
import numpy as np
from datetime import datetime

import timit as tm

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '../data/timit/train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 4,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
TOWER_NAME = 'tower'
NUM_CLASSES = 4

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


def run_training():
    """
    Train the Classy model for a number of steps
    """
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for runway
        images, labels = tm.inputs(FLAGS.batch_size)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        #logits = inference(images)

        # Calculate loss.
        #loss = rw_loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        #train_op = train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            #_, loss_value = sess.run([train_op, loss])
            framet = sess.run([images])[0]
            print(framet)
            duration = time.time() - start_time

            #assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            '''
            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))
            '''
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        sess.close()

'''def run_training():

    with tf.Graph().as_default():
        # Input images and labels.
        frames, labels = inputs(train=True, batch_size=FLAGS.batch_size,
                                num_epochs=FLAGS.num_epochs)

        # The op for initializing the variables.
        #init_op = tf.initialize_all_variables()

        # Create a session for running operations in the Graph.
        with tf.Session() as sess:

            # Initialize the variables (the trained variables and the
            # epoch counter).
            #fr = frames.eval()
            fr = sess.run(labels)
            print('------')
            print(fr)
            print('------')
            print('shape = {0}'.format(fr.shape))
'''

if __name__ == '__main__':
    run_training()
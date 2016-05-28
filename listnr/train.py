#!/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import re
import numpy as np
import tensorflow as tf
from datetime import datetime

import timit as tm
import model as md

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '../data/train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('num_epochs', 30,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('train', True,
                            """Whether to run the training or load a checkpoint.""")


NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 79707
NUM_EXAMPLES_PER_EPOCH_FOR_VAL = 26568
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 5530
NUM_EXAMPLES_PER_EPOCH_FOR_DEV = 10913

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.
LOSS_THRESHOLD = 0.0001

_shutdown = False


def train(total_loss, global_step):
    """Train the Listnr model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = md.add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def run_training():
    """
    Train the Listnr model for a number of steps
    """
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for runway
        # tr_frames_t, tr_labels_t = tm.inputs(FLAGS.batch_size)
        # ts_frames_t, ts_labels_t = tm.inputs(FLAGS.batch_size, train=False)
        # frames, labels = placeholder_inputs()
        frames, labels = tm.inputs(FLAGS.batch_size, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = md.inference(frames)

        # Calculate loss.
        looss = md.loss(logits, labels)

        # calculate accuracy
        accuracy = md.accuracy(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = train(looss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=FLAGS.num_epochs)

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

        # run the training
        steps_per_epoch = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)

        max_steps = FLAGS.num_epochs * steps_per_epoch

        losses_epochs = []
        losses_batches = []
        accuracies_epochs = []
        accuracies_batches = []
        for step in range(max_steps+1):
            start_time = time.time()
            _, loss_value, acc_value = sess.run([train_op, looss, accuracy])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 100 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f, train_acc = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, acc_value, examples_per_sec, sec_per_batch))

                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                losses_batches.append(loss_value)
                accuracies_batches.append(acc_value)

            # Save the model checkpoint periodically.
            if step > 1:
                if (step-1) % steps_per_epoch == 0 or (step + 1) == max_steps or _shutdown:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

                    #accuracies_epochs.append(np.mean(accuracies_batches))
                    #losses_epochs.append(np.mean(losses_batches))

                    # save accuracy and loss
                    np.save(os.path.join(FLAGS.train_dir, 'tr_loss'), np.array(losses_batches))
                    np.save(os.path.join(FLAGS.train_dir, 'tr_accuracy'), np.array(accuracies_batches))
                    print('Saving model: ', (step-1) / steps_per_epoch)
                    accuracies_batches = []
                    losses_batches = []

            if _shutdown:
                break

        print('Listnr training finished!')


def handler(signum, frame):
    global _shutdown
    print('Listnr training shutdown requested! Finalizing...')
    _shutdown = True


def main(argv=None):
    # register signal handlers
    import signal
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    run_training()

if __name__ == '__main__':
    tf.app.run()

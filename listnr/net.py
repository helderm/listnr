#!/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import re
import numpy as np
import tensorflow as tf
from datetime import datetime

import timit as tm

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '../data/timit/train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 2000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_boolean('train', True,
                            """Whether to run the training or load a checkpoint.""")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
TOWER_NAME = 'tower'

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.
LOSS_THRESHOLD = 0.0001

# Constants of the model
NUM_CLASSES = 39
NUM_UNITS_FULL_LAYER = 1000
NUM_FEATURE_MAPS = 150


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

    Returns:
    Variable Tensor
    """
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def inference(frames):
    """Build the Classy inference model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[8, 1, 27, NUM_FEATURE_MAPS],
                                             stddev=1e-4, wd=0.004)
        conv = tf.nn.conv2d(frames, kernel, [1, 2, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [NUM_FEATURE_MAPS], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 6, 1, 1], strides=[1, 2, 1, 1],
                           padding='SAME', name='pool1')

    # local2
    with tf.variable_scope('local2') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool1, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, NUM_UNITS_FULL_LAYER],
                                              stddev=0.04, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_UNITS_FULL_LAYER], tf.constant_initializer(0.1))
        local2 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local2)

    # local3
    with tf.variable_scope('local3') as scope:
        weights = _variable_with_weight_decay('weights', shape=[NUM_UNITS_FULL_LAYER, NUM_UNITS_FULL_LAYER],
                                              stddev=0.04, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_UNITS_FULL_LAYER], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(local2, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [NUM_UNITS_FULL_LAYER, NUM_CLASSES],
                                              stddev=1 / float(NUM_UNITS_FULL_LAYER), wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local3, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

    Returns:
    Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in the Listnr model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


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
    loss_averages_op = _add_loss_summaries(total_loss)

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


def placeholder_inputs():
    """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    frames_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
    # Note that the shapes of the placeholders match the shapes of the full
    # frame and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    frames_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,
                                                           tm.NUM_FILTERS, 1, tm.Total_FEATURES))
    labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))
    return frames_placeholder, labels_placeholder


def fill_feed_dict(frames_pl, labels_pl, sess, train=True):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }

    Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

    Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.
    # images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
    #                                               FLAGS.fake_data)
    frames_t, labels_t = tm.inputs(FLAGS.batch_size, train)
    frames, labels = sess.run([frames_t, labels_t])

    feed_dict = {
        frames_pl: frames,
        labels_pl: labels,
    }
    return feed_dict


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    print(labels)
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


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
        frames, labels = tm.inputs(FLAGS.batch_size)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = inference(frames)

        # Calculate loss.
        looss = loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = train(looss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # get a tensor for evaluating the training
        correct_examples = evaluation(logits, labels)
        # correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        if FLAGS.train:
            # run the training
            prev_loss_value = np.inf
            for step in range(FLAGS.max_steps):
                start_time = time.time()
                #frame_val, label_val = sess.run([frames, labels])
                #print(frame_val)
                #print('-------')
                #print(label_val)
                # feed_dict = fill_feed_dict(frames, labels, sess)

                # frames_val, labels_val = sess.run([tr_frames_t, tr_labels_t])
                #_, loss_value = sess.run([train_op, looss])

                _, loss_value, num_correct_examples = sess.run([train_op, looss, correct_examples])
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                # if np.abs(loss_value - prev_loss_value) < LOSS_THRESHOLD:
                #    format_str = ('%s: Model converged! step %d, loss = %.2f')
                #    print(format_str % (datetime.now(), step, loss_value))
                #    break
                # prev_loss_value = loss_value

                if step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    # frames_val, labels_val = sess.run([ts_frames_t, ts_labels_t])
                    # ts_correct_examples = sess.run(correct_examples, feed_dict={
                    #    frames: frames_val,
                    #    labels: labels_val
                    # })
                    format_str = ('%s: step %d, loss = %.2f, train_acc = %d/%d (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value, num_correct_examples, FLAGS.batch_size,
                                        examples_per_sec, sec_per_batch))

                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
        else:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception('Checkpoint not found!')

        # evaluate against a test set
        # frames, labels = tm.inputs(FLAGS.batch_size, train=False)
        # frames_val, labels_val = sess.run([frames, labels])
        num_correct_examples = sess.run(correct_examples)
        print('test_acc = {0}/{1}'.format(num_correct_examples, FLAGS.batch_size))

        sess.close()


if __name__ == '__main__':
    run_training()

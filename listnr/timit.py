#!/bin/env python
# -*- coding: utf-8 -*-
import os
import json
from pysndfile import sndio
import glob
import numpy as np
import tensorflow as tf
import features.base as ft

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_train_dir', '../data/timit/timit/timit/train/',
                           """Path to the TIMIT data directory.""")
tf.app.flags.DEFINE_string('input_test_dir', '../data/timit/timit/timit/test/',
                           """Path to the TIMIT data directory.""")
tf.app.flags.DEFINE_string('data_dir', '../data/timit/',
                           """Path to the serialized TIMIT dataset.""")
tf.app.flags.DEFINE_string('max_frames', 0,
                           """Max number of frames to import. 0 for no limit.""")

NUM_FILTERS = 40
NUM_FEATURES = 3


#REGIONS = ['dr1']
#NUM_FRAMES = 100243
REGIONS = ['dr1', 'dr2', 'dr3', 'dr4', 'dr5', 'dr6', 'dr7', 'dr8']
NUM_FRAMES = 1236543
#NUM_FRAMES = 451660

_FILENAME_TRAIN = 'train.tfrecords'
_FILENAME_TEST = 'test.tfrecords'

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_delta(frame, deltawindow=4):
    """
    Get the delta of the input frame
    """
    delta = np.zeros(shape=frame.shape)
    for t, coef in enumerate(frame):
        sum_coefs = 0.0
        sum_deltas = 0.0
        for dw in range(1, deltawindow):
            coef_md = frame[t - dw]

            if t + dw < len(frame):
                coef_pd = frame[t + dw]
            else:
                coef_pd = frame[t + dw - len(frame)]

            sum_coefs += dw * (coef_pd - coef_md)
            sum_deltas += 2 * np.power(dw, 2)

        delta[t] = sum_coefs / sum_deltas

    return delta


def _get_words(sfile):
    """
    Get the words info from TIMIT
    :param sfile: word file
    :return: word info
    """
    words = []
    with open(sfile) as f:
        for line in f:
            parts = line.rstrip().split(' ')
            ph = {'start': int(parts[0]),
                  'end': int(parts[1]),
                  'word': parts[2]}

            words.append(ph)

    return words


def _get_phonemes(pfile):
    """
    Get the phonemes info from TIMIT
    :param pfile: phoneme file
    :return: phoneme info
    """
    phonemes = []
    with open(pfile) as f:
        for line in f:
            parts = line.rstrip().split(' ')
            ph = {'start': int(parts[0]),
                  'end': int(parts[1]),
                  'phoneme': parts[2]}

            phonemes.append(ph)

    return phonemes

def _convert_to_record(frame, label, writer):
    """
    Serialize a single frame and label and write it
    to a TFRecord
    :param image: 4D frame tensor
    :param label: 1D label tensor
    :param writer: file writer
    """
    #assert frame.shape[0] == NUM_FILTERS and frame.shape[1] == NUM_FEATURES and frame.shape[2] == 1
    frame_raw = frame.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        # 'height': _float_feature(rows),
        # 'width': _float_feature(cols),
        # 'depth': _float_feature(depth),
        'label': _int64_feature(int(label)),
        'frame_raw': _bytes_feature(frame_raw)}))
    writer.write(example.SerializeToString())


def serialize(train=True):
    """
    Serialize the TIMIT dataset to TFRecords
    :param train:
    :return:
    """

    base_data_path = FLAGS.input_train_dir if train else FLAGS.input_test_dir
    output_path = os.path.join(FLAGS.data_dir, _FILENAME_TRAIN if train else _FILENAME_TEST)

    timit = []
    for region in REGIONS:
        # iterate over all speakers for that region
        region_path = os.path.join(base_data_path, region)
        for speaker_id in os.listdir(region_path):
            speaker_path = os.path.join(region_path, speaker_id)

            # iterate over all utterances for that speaker
            speaker_wavs = glob.glob(speaker_path + '/*.wav')
            for wav in speaker_wavs:

                # get the sound frequencies and sampling rate
                sndobj = sndio.read(wav)
                samplingrate = sndobj[1]
                samples = np.array(sndobj[0]) * np.iinfo(np.int16).max

                # parse the phoneme file
                phonemes = _get_phonemes(wav.replace('.wav', '.phn'))

                # get sentence
                words = _get_words(wav.replace('.wav', '.wrd'))

                timit.append({'filename': wav,
                                'samplingrate': samplingrate,
                                 'phonemes': phonemes,
                                 'words': words,
                                 'gender': speaker_id[0],
                                 'speaker': speaker_id,
                                 'samples': samples})

    phonemes_map = {}
    pho_ctn = 0
    frame_ctn = 0

    frames = np.ndarray(shape=(NUM_FRAMES,NUM_FILTERS, NUM_FEATURES, 1))
    labels = np.ndarray(shape=(NUM_FRAMES))

    # transform the samples into MSFC features
    print('Parsing frames from utterances...')
    for utt in timit:
        samples = utt['samples']
        phonemes = utt['phonemes']

        # extract each phoneme mfsc, delta and delta-delta
        for pho in phonemes:
            if pho['phoneme'] not in phonemes_map:
                phonemes_map[pho['phoneme']] = pho_ctn
                pho_ctn += 1

            # extract the frames for this phonemes only
            pho_idx = phonemes_map[pho['phoneme']]
            pho_samples = samples[pho['start']:pho['end']]

            # get the filterbanks
            mfscs = ft.mfsc(pho_samples, samplerate=utt['samplingrate'], nfilt=NUM_FILTERS)

            # for each frame
            for mfsc in mfscs:
                # add the deltas and delta-deltas for each static frame
                delta = _get_delta(mfsc)
                delta2 = _get_delta(delta)

                # create the new frame representation
                frame = np.ndarray(shape=(NUM_FILTERS, NUM_FEATURES, 1), dtype=np.float32)
                frame[:, 0, :] = mfsc[:,None]
                frame[:, 1, :] = delta[:,None]
                frame[:, 2, :] = delta2[:,None]
                frames[frame_ctn, :, :, :] = frame

                # set the phoneme label
                labels[frame_ctn] = pho_idx

                frame_ctn += 1

                if frame_ctn % 10000 == 0:
                    print('- {0} frames processed...'.format(frame_ctn))

                if FLAGS.max_frames and frame_ctn >= FLAGS.max_frames:
                    break
            if FLAGS.max_frames and frame_ctn >= FLAGS.max_frames:
                break
        if FLAGS.max_frames and frame_ctn >= FLAGS.max_frames:
            break

    print('Finished processing {0} frames!'.format(frame_ctn))
    means = frames.mean(axis=0)
    std = frames.std(axis=0)

    # normalize zero mean and unity variance
    frames = frames - means
    frames = frames / std

    print('Writing', output_path)
    writer = tf.python_io.TFRecordWriter(output_path)

    for i in range(frames.shape[0]):
        # TODO: If I dont do this, the reshape after deserialization gets a wrong size
        frame = np.ndarray(shape=(1, NUM_FILTERS, NUM_FEATURES, 1), dtype=np.float32)
        label = labels[i]
        frame[0, :, :, :] = frames[i]

        _convert_to_record(frame, label, writer)
        if i % 10000 == 0:
            print('- Wrote {0} frames...'.format(i))

    writer.close()

    # save the phoneme mapping file
    with open(os.path.join(FLAGS.data_dir, 'phomap.json'), 'w') as f:
        json.dump(phonemes_map, f, indent=4, sort_keys=True)


def _read_and_decode(filename_queue):
    """
    Reads and deserialize a single example from the queue
    :param filename_queue: files to read
    :return: image, label
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'frame_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    frame = tf.decode_raw(features['frame_raw'], tf.float32)
    frame.set_shape([NUM_FILTERS * NUM_FEATURES * 1])
    frame = tf.reshape(frame, [NUM_FILTERS, NUM_FEATURES, 1])

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    #frame = tf.cast(frame, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return frame, label


def inputs(batch_size, train=True):
    """
    Read frames and labels in shuffled batches
    :param batch_size: size of the batch
    :return: images 4D tensor, labels 1D tensor
    """
    filename = os.path.join(FLAGS.data_dir, _FILENAME_TRAIN if train else _FILENAME_TEST)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename])

        # Even when reading in multiple threads, share the filename
        # queue.
        frame, label = _read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, sparse_labels = tf.train.shuffle_batch(
            [frame, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

    return images, sparse_labels

if __name__ == '__main__':
    serialize()
    #serialize(train=False)
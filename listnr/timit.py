#!/bin/env python
# -*- coding: utf-8 -*-
import os
import operator
import json
from pysndfile import sndio
import glob
import numpy as np
import tensorflow as tf
import features.base as ft

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_train_dir', '../data/timit/timit/timit/train/',
                           """Path to the TIMIT data directory.""")
tf.app.flags.DEFINE_string('input_test_dir', '../data/timit/timit/timit/core_test/',
                           """Path to the TIMIT data directory.""")
tf.app.flags.DEFINE_string('data_dir', '../data/timit/train/',
                           """Path to the serialized TIMIT dataset.""")
tf.app.flags.DEFINE_integer('data_files', 12,
                            """ Number of separate data files to have.""")

NUM_FILTERS = 40
NUM_FEATURES = 3
FrameSize = 9
Total_FEATURES = NUM_FEATURES * FrameSize

#REGIONS = ['dr1']
#TRN_NUM_FRAMES = 100243
REGIONS = ['dr1', 'dr2', 'dr3', 'dr4', 'dr5', 'dr6', 'dr7', 'dr8']
TRN_NUM_FRAMES = 1236543
TST_NUM_FRAMES = 451660

_FILENAME_TRAIN = 'train.tfrecords'
_FILENAME_TEST = 'test.tfrecords'
_FILENAME_DEV = 'dev.tfrecords'

class2pho = {
    'aa': {'idx': 0, 'pho': ['aa', 'ao']},
    'ah': {'idx': 1, 'pho': ['ah', 'ax', 'ax-h']},
    'er': { 'idx': 2, 'pho': ['er', 'axr']},
    'hh': {'idx': 3, 'pho': ['hh', 'hv']},
    'ih': {'idx': 4, 'pho': ['ih', 'ix']},
    'l': {'idx': 5, 'pho': ['l', 'el']},
    'm': {'idx': 6, 'pho': ['m', 'em']},
    'n': {'idx': 7, 'pho': ['n', 'en', 'nx']},
    'ng': {'idx': 8, 'pho': ['ng', 'eng']},
    'sh': {'idx': 9, 'pho': ['sh', 'zh']},
    'uw': {'idx': 10, 'pho': ['uw', 'ux']},
    'sil': {'idx': 11, 'pho': ['pcl', 'tcl', 'kcl', 'bcl', 'dcl', 'gcl',
                               'h#', 'pau', 'epi']},
}


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_delta(frame, deltawindow=2):
    """
    Get the delta of the input frame
    """
    delta = np.zeros(shape=frame.shape)
    for t, coef in enumerate(frame):

        if t < deltawindow:
            delta[t] = frame[t+1] - frame[t]
            continue
        if t >= len(frame) - deltawindow:
            delta[t] = frame[t] - frame[t-1]
            continue

        sum_coefs = 0.0
        sum_deltas = 0.0
        for dw in range(1, deltawindow):

            coef_md = frame[t - dw]
            coef_pd = frame[t + dw]

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
    global class2pho

    phonemes = []
    with open(pfile) as f:
        for line in f:
            parts = line.rstrip().split(' ')
            phoneme = parts[2]
            if phoneme == 'q':
                continue

            # search for the phoneme in th
            found = False
            if phoneme not in class2pho:
                for phm, val in class2pho.items():
                    if phoneme in val['pho']:
                        found = True
                        break
                if found:
                    phoneme = phm
                else:
                    idx = max(class2pho.values(), key=lambda x: x['idx'])['idx']
                    #idx = max(class2pho.iteritems(), key=operator.itemgetter(1))[1]
                    class2pho[phoneme] = {'idx': idx+1, 'pho': [phoneme]}

            phd = {'start': int(parts[0]),
                  'end': int(parts[1]),
                  'phoneme': phoneme}

            phonemes.append(phd)

    return phonemes


def _convert_to_record(frame, label, writer):
    """
    Serialize a single frame and label and write it
    to a TFRecord
    :param image: 4D frame tensor
    :param label: 1D label tensor
    :param writer: file writer
    """
    assert frame.shape[1] == NUM_FILTERS and frame.shape[3] == Total_FEATURES and frame.shape[2] == 1
    frame_raw = frame.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
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
    num_frames = TRN_NUM_FRAMES/FrameSize if train else TST_NUM_FRAMES/FrameSize

    timit = []
    print('Parsing .wav files...')
    for region in REGIONS:
        # iterate over all speakers for that region
        region_path = os.path.join(base_data_path, region)
        for speaker_id in os.listdir(region_path):
            speaker_path = os.path.join(region_path, speaker_id)

            # iterate over all utterances for that speaker
            speaker_wavs = glob.glob(speaker_path + '/*.wav')
            for wav in speaker_wavs:
                if "sa" not in wav:
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

    frame_ctn = 0

    frames = np.ndarray(shape=(num_frames,NUM_FILTERS,1, Total_FEATURES))
    labels = np.ndarray(shape=(num_frames))

    # transform the samples into MSFC features
    print('Parsing frames from utterances...')

    # adding the counter for fix-frames input
    count = 0
    input_sample = np.ndarray(shape=(NUM_FILTERS, 1, Total_FEATURES), dtype=np.float32)
    label_list = []

    for utt in timit:
        samples = utt['samples']
        phonemes = utt['phonemes']

        # extract each phoneme mfsc, delta and delta-delta
        for pho in phonemes:
            # extract the frames for this phonemes only
            pho_idx = class2pho[pho['phoneme']]['idx']
            pho_samples = samples[pho['start']:pho['end']]

            # get the filterbanks
            mfscs = ft.mfsc(pho_samples, samplerate=utt['samplingrate'], nfilt=NUM_FILTERS)

            # for each frame
            for mfsc in mfscs:
                # add the deltas and delta-deltas for each static frame
                delta = _get_delta(mfsc)
                delta2 = _get_delta(delta)

                # create the new frame representation
                frame = np.ndarray(shape=(NUM_FILTERS, 1, NUM_FEATURES), dtype=np.float32)
                frame[:, :, 0] = mfsc[:,None]
                frame[:, :, 1] = delta[:,None]
                frame[:, :, 2] = delta2[:,None]
                
                input_sample[:, :, 3 * count: 3 * (count + 1)] = frame
                label_list.append(pho_idx)
                count += 1
                if count == 9:
                    count = 0
                    frames[frame_ctn, :, :, :] = input_sample
                    #print(label_list)
                    #print(Counter(label_list).most_common()[0][0])
                    #labels[frame_ctn] = Counter(label_list).most_common()[0][0]
                    #print(label_list[4])
                    labels[frame_ctn] = label_list[4]
                    frame_ctn += 1
                    #print('Finish ', frame_ctn)
                    input_sample = np.ndarray(shape=(NUM_FILTERS,1, Total_FEATURES), dtype=np.float32)
                    label_list.clear()

                    if frame_ctn % 1000 == 0:
                        print('- {0} frames processed...'.format(frame_ctn))

    frames = frames[0:frame_ctn, :, :, :]
    labels = labels[0:frame_ctn]

    print('Finished processing {0} frames!'.format(frame_ctn))
    means = frames.mean(axis=0)
    std = frames.std(axis=0)

    # normalize zero mean and unity variance
    frames = frames - means
    frames = frames / std

    num_examples_per_file = int(frames.shape[0] / FLAGS.data_files)
    file_idx = 0
    #filename = os.path.join(FLAGS.data_dir, output_path + str(file_idx))
    filename = output_path + str(file_idx)
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for i in range(frames.shape[0]):
        frame = np.ndarray(shape=(1, NUM_FILTERS, 1,  Total_FEATURES), dtype=np.float32)
        label = labels[i]
        frame[0, :, :, :] = frames[i]

        _convert_to_record(frame, label, writer)
        if i % 1000 == 0:
            print('- Wrote {0}/{1} frames...'.format(i, frames.shape[0]))

        if (i + 1) % num_examples_per_file == 0 and i+1 < frames.shape[0]:
            writer.close()
            file_idx += 1
            #filename = os.path.join(FLAGS.data_dir, output_path + str(file_idx))
            filename = output_path + str(file_idx)
            print('Writing', filename)
            writer = tf.python_io.TFRecordWriter(filename)

    writer.close()

    # save the phoneme mapping file
    with open(os.path.join(FLAGS.data_dir, 'phon_tr.json' if train else 'phon_tst.json'), 'w') as f:
        json.dump(class2pho, f, indent=4, sort_keys=True)


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
    frame.set_shape([NUM_FILTERS * 1 * Total_FEATURES])
    frame = tf.reshape(frame, [NUM_FILTERS, 1, Total_FEATURES])

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    #frame = tf.cast(frame, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return frame, label


def inputs(batch_size, num_examples_epoch, type='train', num_epochs=None, shuffle=True):
    """
    Read frames and labels in shuffled batches
    :param batch_size: size of the batch
    :return: images 4D tensor, labels 1D tensor
    """

    if type == 'train':
        filenames = [os.path.join(FLAGS.data_dir, _FILENAME_TRAIN + str(i))
                 for i in range(FLAGS.data_files)]
    #            for i in range(1)]
    elif type == 'test':
        filenames = [os.path.join(FLAGS.data_dir, _FILENAME_TEST + str(i))
                     for i in range(FLAGS.data_files)]
    elif type == 'dev':
        filenames = [os.path.join(FLAGS.data_dir, _FILENAME_DEV + str(i))
                     for i in range(FLAGS.data_files)]
    else:
        raise Exception('Unkwown input type')

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle)

        # Even when reading in multiple threads, share the filename
        # queue.
        frame, label = _read_and_decode(filename_queue)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_epoch *
                                 min_fraction_of_examples_in_queue)

        if shuffle:
            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            images_batch, labels_batch = tf.train.shuffle_batch(
                [frame, label], batch_size=batch_size, num_threads=2,
                capacity=min_queue_examples + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=min_queue_examples)
        else:
            images_batch, labels_batch = tf.train.batch(
                [frame, label],
                batch_size=batch_size,
                num_threads=2,
                capacity=min_queue_examples + 3 * batch_size)

    return images_batch, labels_batch

if __name__ == '__main__':
    #serialize()
    serialize(train=False)

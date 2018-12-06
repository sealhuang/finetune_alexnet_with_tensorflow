
"""Containes a helper class for image input pipelines in tensorflow."""

import os
import tensorflow as tf
import numpy as np
from random import shuffle as list_shuffle

from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

landmark_stats = './bazhong/norm_landmark_stats.npz'
lmstats = np.load(landmark_stats)
LMMEAN = tf.constant(lmstats['meanall'], dtype=tf.float32)
LMSTD = tf.constant(lmstats['stdall'], dtype=tf.float32)

class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True,
                 buffer_size=1000):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.txt_file = txt_file
        self.num_classes = num_classes

        # retrieve the data from the text file
        self._read_txt_file(shuffle)

        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.landmarks = convert_to_tensor(self.landmarks, dtype=dtypes.float32)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.landmarks, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_threads=8,
                      output_buffer_size=100*batch_size)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_threads=8,
                      output_buffer_size=100*batch_size)
        elif mode == 'test':
            data = data.map(self._parse_function_test, num_threads=8,
                      output_buffer_size=100*batch_size)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _read_txt_file(self, shuffle=False):
        """Read the content of the text file and store it into lists."""
        self.landmarks = []
        self.labels = []
        current_dir = os.getcwd()
        # randomly sample the data to balancing various categories
        start_val = 60
        #end_val = 140
        end_val = 160
        val_interval = (end_val - start_val) / self.num_classes
        #num_per_bin = 50 * (8 / self.num_classes)
        num_per_bin = 100 * (10 / self.num_classes)
        bin_mark = range(start_val, end_val+1, val_interval)
        num_count = [0] * (len(bin_mark)-1)
        bn = [0, 0]
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            # shuffle the lines
            if shuffle:
                list_shuffle(lines)
            for line in lines:
                items = line.strip().split(',')
                #for mi in range(len(bin_mark)):
                #    if v<bin_mark[mi]:
                #        if num_count[mi-1]<num_per_bin:
                #            self.img_paths.append(p)
                #            self.labels.append(mi-1)
                #            num_count[mi-1] += 1
                #            #print v, mi-1
                #        break
                v = float(items[1])
                if v<85 and bn[0]<1296:
                    lms = [float(items[2+i]) for i in range(144)]
                    self.landmarks.append(lms)
                    self.labels.append(0)
                    bn[0] += 1
                elif v>=115 and bn[1]<1296:
                    lms = [float(items[2+i]) for i in range(144)]
                    self.landmarks.append(lms)
                    self.labels.append(1)
                    bn[1] += 1
                else:
                    pass
                #v = items[0]
                #if int(v[16])%2 and bn[0]<1296:
                #    lms = [float(items[2+i]) for i in range(144)]
                #    self.landmarks.append(lms)
                #    self.labels.append(0)
                #    bn[0] += 1
                #elif (not int(v[16])%2) and bn[1]<1296:
                #    lms = [float(items[2+i]) for i in range(144)]
                #    self.landmarks.append(lms)
                #    self.labels.append(1)
                #    bn[1] += 1
                #else:
                #    pass

        print 'Load %s samples'%(len(self.labels))
        print 'data dist',
        #print num_count
        print bn

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        landmark = self.landmarks
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.landmarks = []
        self.labels = []
        for i in permutation:
            self.landmarks.append(landmark[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, landmark, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        norm_lm = tf.subtract(landmark, LMMEAN)
        norm_lm = tf.div(norm_lm, LMSTD)

        return norm_lm, one_hot

    def _parse_function_inference(self, landmark, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)
        
        norm_lm = tf.subtract(landmark, LMMEAN)
        norm_lm = tf.div(norm_lm, LMSTD)

        return norm_lm, one_hot
    
    def _parse_function_test(self, filename, label):
        """Input parser for samples of the test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        #img_decoded = tf.image.decode_png(img_string, channels=3)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_aligned = tf.image.resize_image_with_crop_or_pad(img_decoded,
                                                             380, 380)
        img_resized = tf.image.resize_images(img_aligned, [227, 227])
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot

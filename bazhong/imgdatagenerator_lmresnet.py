# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""Containes a helper class for image input pipelines in tensorflow."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset


class ImageDataGenerator(object):
    """Wrapper class around the Tensorflows dataset pipeline.
    
    """
    def __init__(self, lm_dist, label_list, mode, batch_size, num_classes,
                 shuffle=True, buffer_size=1000):
        """Create a new ImageDataGenerator.

        Recieves a `img_list` which each element is a path string to an
        image, and a `label_list` which each element is an integer referring
        to the class number. Using this data, this class will create
        TensrFlow datasets, that can be used to train e.g. a convolutional
        neural network.

        Args:
            img_list: List of paths to the image.
            label_list: List of class number.
            mode: Either 'training', 'validation', or 'test'. Depending
                on this value, different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.lm_dist = lm_dist
        self.label_list = label_list
        self.num_classes = num_classes

        # number of samples in the dataset
        self.data_size = len(self.label_list)

        # initial shuffling of the image and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.dists = tf.convert_to_tensor(self.lm_dist, dtype=tf.float32)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.dists, self.labels))

        # distinguish between train, infer, and test data. when calling the
        # parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train,
                            num_parallel_calls=8)
            data = data.prefetch(batch_size*10)
        elif mode == 'inference':
            data = data.map(self._parse_function_inference,
                            num_parallel_calls=8)
            data = data.prefetch(batch_size*10)
        elif mode == 'test':
            data = data.map(self._parse_function_test,
                            num_parallel_calls=8)
        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        lm_dist = self.lm_dist
        labels = self.label_list
        self.lm_dist = np.zeros_like(lm_dist)
        self.label_list = []
        c = 0
        for i in np.random.permutation(self.data_size):
            self.lm_dist[c, :, :] = lm_dist[i, :, :]
            c += 1
            self.label_list.append(labels[i])

    def _parse_function_train(self, lm_dist, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)
        new_lm_dist = tf.expand_dims(lm_dist, -1)

        return new_lm_dist, one_hot

    def _parse_function_inference(self, lm_dist, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)
        new_lm_dist = tf.expand_dims(lm_dist, -1)

        return new_lm_dist, one_hot
    
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


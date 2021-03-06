# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""Containes a helper class for image input pipelines in tensorflow."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset

IMG_MEAN = tf.constant([171.0, 148.0, 139.0], dtype=tf.float32)
IMG_STD = tf.constant([32.0, 25.0, 25.0], dtype=tf.float32)

lmstats = np.load('./data/norm_landmark_stats.npz')
LM_MEAN = tf.constant(lmstats['meanall'], dtype=tf.float32)
LM_STD = tf.constant(lmstats['stdall'], dtype=tf.float32)

class AlexDataGenerator(object):
    """Wrapper class around the Tensorflows dataset pipeline.
    
    """
    def __init__(self, img_list, label_list, mode, batch_size, num_classes,
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
        self.img_list = img_list
        self.label_list = label_list
        self.num_classes = num_classes

        # number of samples in the dataset
        self.data_size = len(self.label_list)

        # initial shuffling of the image and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.imgs = tf.convert_to_tensor(self.img_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.imgs, self.labels))

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
        data = data.prefetch(1)

        self.data = data

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        imgs = self.img_list
        labels = self.label_list
        self.img_list = []
        self.label_list = []
        for i in np.random.permutation(self.data_size):
            self.img_list.append(imgs[i])
            self.label_list.append(labels[i])

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # radnomly load an image and preprocess the image
        null_flag = tf.equal(filename, tf.constant(['null']))
        null_size = tf.reduce_sum(tf.cast(null_flag, tf.int32))
        non_null_size = tf.size(filename) - null_size
        rand_idx = tf.random_uniform([1], 0, non_null_size, dtype=tf.int32)[0]
        img_string = tf.read_file(filename[rand_idx])
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [250, 250])
        """
        Dataaugmentation comes here.
        """
        img_distorted = tf.random_crop(img_resized, [227, 227, 3])
        #img_distorted = tf.image.random_flip_left_right(img_distorted)
        img_centered = tf.subtract(img_distorted, IMG_MEAN)
        img_centered = tf.div(img_centered, IMG_STD)
        img_centered = tf.image.random_brightness(img_centered, max_delta=0.3)

        img_out = img_centered
        #img_out = tf.image.rgb_to_grayscale(img_centered)

        return img_out, one_hot

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        # radnomly load an image and preprocess the image
        null_flag = tf.equal(filename, tf.constant(['null']))
        null_size = tf.reduce_sum(tf.cast(null_flag, tf.int32))
        non_null_size = tf.size(filename) - null_size
        rand_idx = tf.random_uniform([1], 0, non_null_size, dtype=tf.int32)[0]
        img_string = tf.read_file(filename[rand_idx])
        #img_string = tf.read_file(filename[0])
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, IMG_MEAN)
        img_centered = tf.div(img_centered, IMG_STD)

        img_out = img_centered
        #img_out = tf.image.rgb_to_grayscale(img_centered)

        return img_out, one_hot
    
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

class ResnetDataGenerator(object):
    """Wrapper class around the Tensorflows dataset pipeline.
    
    """
    def __init__(self, img_list, label_list, mode, batch_size, num_classes,
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
        self.img_list = img_list
        self.label_list = label_list
        self.num_classes = num_classes

        # number of samples in the dataset
        self.data_size = len(self.label_list)

        # initial shuffling of the image and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.imgs = tf.convert_to_tensor(self.img_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.imgs, self.labels))

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
        data = data.prefetch(1)

        self.data = data

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        imgs = self.img_list
        labels = self.label_list
        self.img_list = []
        self.label_list = []
        for i in np.random.permutation(self.data_size):
            self.img_list.append(imgs[i])
            self.label_list.append(labels[i])

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # radnomly load an image and preprocess the image
        null_flag = tf.equal(filename, tf.constant(['null']))
        null_size = tf.reduce_sum(tf.cast(null_flag, tf.int32))
        non_null_size = tf.size(filename) - null_size
        rand_idx = tf.random_uniform([1], 0, non_null_size, dtype=tf.int32)[0]
        img_string = tf.read_file(filename[rand_idx])
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [250, 250])
        """
        Dataaugmentation comes here.
        """
        img_distorted = tf.random_crop(img_resized, [224, 224, 3])
        #img_distorted = tf.image.random_flip_left_right(img_distorted)
        img_centered = tf.subtract(img_distorted, IMG_MEAN)
        img_centered = tf.div(img_centered, IMG_STD)
        img_centered = tf.image.random_brightness(img_centered, max_delta=0.3)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        # radnomly load an image and preprocess the image
        null_flag = tf.equal(filename, tf.constant(['null']))
        null_size = tf.reduce_sum(tf.cast(null_flag, tf.int32))
        non_null_size = tf.size(filename) - null_size
        rand_idx = tf.random_uniform([1], 0, non_null_size, dtype=tf.int32)[0]
        img_string = tf.read_file(filename[rand_idx])
        #img_string = tf.read_file(filename[0])
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [224, 224])
        img_centered = tf.subtract(img_resized, IMG_MEAN)
        img_centered = tf.div(img_centered, IMG_STD)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot
    
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

class LandmarkDataGenerator(object):
    """Wrapper class around the Tensorflows dataset pipeline.
    
    """
    def __init__(self, landmark_list, label_list, mode, batch_size, num_classes,
                 shuffle=True, buffer_size=1000):
        """Create a new ImageDataGenerator.

        Recieves a `landmark_list` which each element is a list of landmarks
        to a face, and a `label_list` which each element is an integer referring
        to the class number. Using this data, this class will create
        TensrFlow datasets, that can be used to train e.g. a convolutional
        neural network.

        Args:
            landmark_list: List of landmarks of the face.
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
        self.landmark_list = landmark_list
        self.label_list = label_list
        self.num_classes = num_classes

        # number of samples in the dataset
        self.data_size = len(self.label_list)

        # initial shuffling of the image and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.landmarks = tf.convert_to_tensor(self.landmark_list,
                                              dtype=tf.float32)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.landmarks, self.labels))

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
        data = data.prefetch(1)

        self.data = data

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        landmarks = self.landmark_list
        labels = self.label_list
        self.landmark_list = []
        self.label_list = []
        for i in np.random.permutation(self.data_size):
            self.landmark_list.append(landmarks[i])
            self.label_list.append(labels[i])

    def _parse_function_train(self, landmark, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        norm_lm = tf.subtract(landmark, LM_MEAN)
        norm_lm = tf.div(norm_lm, LM_STD)

        return norm_lm, one_hot

    def _parse_function_inference(self, landmark, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        norm_lm = tf.subtract(landmark, LM_MEAN)
        norm_lm = tf.div(norm_lm, LM_STD)

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

class MBTIDataGenerator(object):
    """Wrapper class around the Tensorflows dataset pipeline.
    
    """
    def __init__(self, img_list, label_list, mode, batch_size, num_classes,
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
        self.img_list = img_list
        self.label_list = label_list
        self.num_classes = num_classes

        # number of samples in the dataset
        self.data_size = len(self.label_list)

        # initial shuffling of the image and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.imgs = tf.convert_to_tensor(self.img_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.imgs, self.labels))

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
        data = data.prefetch(1)

        self.data = data

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        imgs = self.img_list
        labels = self.label_list
        self.img_list = []
        self.label_list = []
        for i in np.random.permutation(self.data_size):
            self.img_list.append(imgs[i])
            self.label_list.append(labels[i])

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load image and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [250, 250])
        """
        Dataaugmentation comes here.
        """
        img_distorted = tf.random_crop(img_resized, [227, 227, 3])
        #img_distorted = tf.image.random_flip_left_right(img_distorted)
        img_centered = tf.subtract(img_distorted, IMG_MEAN)
        img_centered = tf.div(img_centered, IMG_STD)
        img_centered = tf.image.random_brightness(img_centered, max_delta=0.3)

        img_out = img_centered
        #img_out = tf.image.rgb_to_grayscale(img_centered)

        return img_out, one_hot

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, IMG_MEAN)
        img_centered = tf.div(img_centered, IMG_STD)

        img_out = img_centered
        #img_out = tf.image.rgb_to_grayscale(img_centered)

        return img_out, one_hot
    
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


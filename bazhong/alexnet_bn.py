# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import tensorflow as tf
import numpy as np


class AlexNetBN(object):
    """Implementation of the AlexNet."""

    def __init__(self, x, keep_prob, num_classes, is_train):
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
        """
        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.IS_TRAIN = is_train

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Create the network graph."""
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = conv(self.X, 11, 11, 48, 4, 4, padding='VALID', name='conv1',
                     is_train=self.IS_TRAIN)
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        
        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 64, 1, 1, name='conv2',
                     is_train=self.IS_TRAIN)
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        
        # 3rd Layer: Conv (w ReLu)
        #conv3 = conv(pool2, 3, 3, 64, 1, 1, name='conv3',
        #             is_train=self.IS_TRAIN)

        # 4th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv4 = conv(pool2, 3, 3, 96, 1, 1, name='conv4',
                     is_train=self.IS_TRAIN)
        pool4 = max_pool(conv4, 13, 13, 1, 1, padding='VALID', name='pool4')
        pool4 = tf.reshape(pool4, [-1, 96])
        #pool4 = tf.reduce_mean(conv4, axis=[1, 2])

        fc5 = fc(pool4, 96, 32, is_train=self.IS_TRAIN, relu=True,
                 name='fc5')
        #dropout5 = dropout(fc5, self.KEEP_PROB)
        #fc6 = fc(fc5, 64, 16, is_train=self.IS_TRAIN, relu=True,
        #         name='fc6')
        fc7 = fc(fc5, 32, self.NUM_CLASSES, is_train=self.IS_TRAIN, relu=False,
                 name='fc7')
        self.logits = fc7

    def load_initial_weights(self, session):
        """Load weights from file into network."""
        pass

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         is_train, padding='SAME'):
    """Create a convolution layer."""
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    conv = convolve(x, weights)
 
    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # apply batch normalization
    bn = tf.layers.batch_normalization(bias, axis=-1, training=is_train)

    # Apply relu function
    relu = tf.nn.relu(bn, name=scope.name)

    return relu

def fc(x, num_in, num_out, name, is_train,  relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    

    if relu:
        # apply batch normalization
        bn = tf.layers.batch_normalization(act, axis=-1, training=is_train)
        # Apply ReLu non linearity
        relu = tf.nn.relu(bn)
        return relu
    else:
        return act

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)

def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)

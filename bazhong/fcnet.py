import tensorflow as tf
import numpy as np


class FCNet(object):
    """Implementation of the all-fc Net."""

    def __init__(self, x, keep_prob, num_classes, is_train):
        """Create the graph of the FCNet model.

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
        #dropout1 = dropout(self.X, self.KEEP_PROB)
        fc1 = fc(self.X, 144, 256, relu=True, name='fc1')
        fc2 = fc(fc1, 256, 256, relu=True, name='fc2')
        dropout2 = dropout(fc2, self.KEEP_PROB)
        fc3 = fc(dropout2, 256, 128, relu=False, name='fc3')
        bn3 = tf.layers.batch_normalization(fc3, axis=1,
                                            training=self.IS_TRAIN, name='bn3')
        relu3 = tf.nn.relu(bn3)
        fc4 = fc(relu3, 128, self.NUM_CLASSES, relu=False, name='fc4')
        self.logits = fc4


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act

def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)

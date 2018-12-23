# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import tensorflow as tf
import numpy as np


class ResNet(object):
    """Implementation of 18-layer ResNet."""
    def __init__(self, images, scalars, num_classes, skip_layer, is_train,
                 keep_prob, weights_path='DEFAULT'):
        """Create the graph of ResNet-18 model.

        Args:
            images: Placeholder for the input tensor.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained
                from scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code

        """
        # Parse input arguments into variables
        self.X = images
        self.SCALAR = scalars
        self.KEEP_PROB = keep_prob
        self.NUM_CLASSES = num_classes
        self.SKIP_LAYER = skip_layer
        # pretrained parameters config
        if weights_path=='DEFAULT':
            self.WEIGHTS_PATH = 'resnet18_paras.npz'
        else:
            self.WEIGHTS_PATH = weights_path

        self.is_train = is_train
        self._counted_scope = []
        self._flops = 0
        self._weights = 0

        self.create()

    def create(self):
        """Create the network graph."""
        print('Building model')
        # filters = [128, 128, 256, 512, 1024]
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]

        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1'):
            x = self._conv(self.X, kernels[0], filters[0], strides[0])
            x = self._bn(x)
            x = self._relu(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # conv2_x
        x = self._residual_block(x, name='conv2_1')
        x = self._residual_block(x, name='conv2_2')

        # conv3_x
        x = self._residual_block_first(x, filters[2], strides[2],name='conv3_1')
        x = self._residual_block(x, name='conv3_2')

        # conv4_x
        x = self._residual_block_first(x, filters[3], strides[3],name='conv4_1')
        x = self._residual_block(x, name='conv4_2')

        # conv5_x
        x = self._residual_block_first(x, filters[4], strides[4],name='conv5_1')
        x = self._residual_block(x, name='conv5_2')

        # Logit
        with tf.variable_scope('logits') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = tf.reduce_mean(x, [1, 2])
            x = tf.concat([x, tf.expand_dims(self.SCALAR, 1)], axis=1)
            x = self._fc(x, 256, name='fc1')
            x = self._relu(x)
            x = tf.nn.dropout(x, self.KEEP_PROB)
            x = self._fc(x, 128, name='fc2')
            x = self._relu(x)
            x = self._fc(x, self.NUM_CLASSES)
        
        # model output
        self.logits = x

    def load_initial_weights(self, session):
        """Load pretrained weights from file into network."""
        #TODO
        pass

    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1],
                                              [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides,
                                      name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, out_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x

    def _residual_block(self, x, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv(x, 3, num_channel, 1, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, num_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')

            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x


    # Helper functions(counts FLOPs and number of weights)
    def _conv(self, x, filter_size, out_channel, stride,pad="SAME",name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = conv_(x, filter_size, out_channel, stride, pad, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * \
            filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _fc(self, x, out_dim, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = fc_(x, out_dim, name)
        f = 2 * (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _bn(self, x, name="bn"):
        x = bn_(x, self.is_train, name)
        # f = 8 * self._get_data_size(x)
        # w = 4 * x.get_shape().as_list()[-1]
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, w)
        return x

    def _relu(self, x, name="relu"):
        x = relu_(x, 0.0, name)
        # f = self._get_data_size(x)
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, 0)
        return x

    def _get_data_size(self, x):
        return np.prod(x.get_shape().as_list()[1:])

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)


## TensorFlow helper functions
def conv_(x, filter_size, out_channel, strides, pad='SAME', name='conv'):
    in_shape = x.get_shape()
    with tf.variable_scope(name):
        # Main operation: conv2d
        with tf.device('/CPU:0'):
            kernel = tf.get_variable('kernel',
                        [filter_size, filter_size, in_shape[3], out_channel],
                        tf.float32,
                        initializer=tf.random_normal_initializer(
                            stddev=np.sqrt(2.0/filter_size/filter_size/out_channel)))
        if kernel not in tf.get_collection('WEIGHT_DECAY'):
            tf.add_to_collection('WEIGHT_DECAY', kernel)
        conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], pad)
    return conv

def bn_(x, is_train, name='bn'):
    moving_average_decay = 0.9
    with tf.variable_scope(name):
        decay = moving_average_decay
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        with tf.device('/CPU:0'):
            mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer(), trainable=False)
            sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer(), trainable=False)
            beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer())
        # BN when training
        update = 1.0 - decay
        update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_sigma = sigma.assign_sub(update*(sigma - batch_var))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

        mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                            lambda: (mu, sigma))
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return bn

def relu_(x, leakness=0.0, name=None):
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x*leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')

def fc_(x, out_dim, name='fc'):
    with tf.variable_scope(name):
        # Main operation: fc
        with tf.device('/CPU:0'):
            w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                        tf.float32, initializer=tf.random_normal_initializer(
                                stddev=np.sqrt(1.0/out_dim)))
            b = tf.get_variable('biases', [out_dim], tf.float32,
                                initializer=tf.constant_initializer(0.0))
        if w not in tf.get_collection('WEIGHT_DECAY'):
            tf.add_to_collection('WEIGHT_DECAY', w)
        fc = tf.nn.bias_add(tf.matmul(x, w), b)
    return fc


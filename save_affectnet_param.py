# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""Script to save AffectNet model parameters using Tensorflow."""

import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf

from salexnet import AlexNet
from datetime import datetime

"""
Configuration Part.
"""
current_dir = os.getcwd()

# Network params
num_classes = 4
batch_size = 32

# Path to store model checkpoints
checkpoint_path = os.path.join(current_dir, 'log', 'checkpoints')

"""
Main Part of the finetuning Script.
"""

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, [])

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    # Load the pretrained weights into the model
    saver.restore(sess, os.path.join(checkpoint_path, 'sel_model_epoch44.ckpt'))
    # get graph
    graph = sess.graph

    conv1 = []
    conv1.append(graph.get_tensor_by_name('conv1/weights:0').eval())
    conv1.append(graph.get_tensor_by_name('conv1/biases:0').eval())
    conv2 = []
    conv2.append(graph.get_tensor_by_name('conv2/weights:0').eval())
    conv2.append(graph.get_tensor_by_name('conv2/biases:0').eval())
    conv3 = []
    conv3.append(graph.get_tensor_by_name('conv3/weights:0').eval())
    conv3.append(graph.get_tensor_by_name('conv3/biases:0').eval())
    conv4 = []
    conv4.append(graph.get_tensor_by_name('conv4/weights:0').eval())
    conv4.append(graph.get_tensor_by_name('conv4/biases:0').eval())
    conv5 = []
    conv5.append(graph.get_tensor_by_name('conv5/weights:0').eval())
    conv5.append(graph.get_tensor_by_name('conv5/biases:0').eval())
    fc6 = []
    fc6.append(graph.get_tensor_by_name('fc6/weights:0').eval())
    fc6.append(graph.get_tensor_by_name('fc6/biases:0').eval())
    fc7 = []
    fc7.append(graph.get_tensor_by_name('fc7/weights:0').eval())
    fc7.append(graph.get_tensor_by_name('fc7/biases:0').eval())

np.savez('affectnet_params', conv1=conv1, conv2=conv2, conv3=conv3, conv4=conv4,
                             conv5=conv5, fc6=fc6, fc7=fc7)


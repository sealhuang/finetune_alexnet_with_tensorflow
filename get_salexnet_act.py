# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""Script to get image activation from sAlexNet."""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf

from salexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator

"""
Configuration Part.
"""
current_dir = os.getcwd()
img_list_file = 'S1_stimuli_1.txt'

# Path to the textfiles for test set
test_file = os.path.join(current_dir, 'emoImg', 'stimseq', img_list_file)

# Network params
num_classes = 4
batch_size = 16

# Path to store model checkpoints
checkpoint_path = os.path.join(current_dir, 'log', 'checkpoints')

"""
Main Part of the finetuning Script.
"""

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    test_data = ImageDataGenerator(test_file,
                                   mode='test',
                                   batch_size=batch_size,
                                   num_classes=num_classes,
                                   shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(test_data.data.output_types,
                                       test_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the iterators
testing_init_op = iterator.make_initializer(test_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, [])

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of test steps per epoch
test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))

pred = np.array([])

# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    # Load the pretrained weights into the model
    saver.restore(sess, os.path.join(checkpoint_path, 'sel_model_epoch44.ckpt'))
    graph = sess.graph
    pool5_1_out = graph.get_tensor_bu_name('Conv2D_6:0')
    pool5_2_out = graph.get_tensor_bu_name('Conv2D_7:0')

    # Test the model on the entire validation set
    print("{} Start test".format(datetime.now()))
    sess.run(testing_init_op)
    for _ in range(test_batches_per_epoch):
        img_batch, label_batch = sess.run(next_batch)
        p51, p52 = sess.run([pool5_1_out, pool5_2_out],
                            feed_dict={x: img_batch,
                                       y: label_batch,
                                       keep_prob: 1.})
        #pred = np.concatenate((pred, pred_y))
        print p51.shape
        print p52.shape



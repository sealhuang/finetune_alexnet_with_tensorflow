# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""Script to test AlexNet using Tensorflow on ImageNet task."""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import tensorflow as tf

from alexnet import AlexNet
from caffe_classes import class_names

# mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

current_dir = os.getcwd()
image_dir = os.path.join(current_dir, 'images')

# get list of all images
img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
             if f.endswith('.jpeg')]

# load all images
imgs = []
for f in img_files:
    print f
    imgs.append(cv2.imread(f))

# placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)
# create model with default config
model = AlexNet(x, keep_prob, 1000, [])
# define activation of last layer as score
score = model.fc8
# create op to calculate softmax
softmax = tf.nn.softmax(score)

with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Load the pretrained weights into the model
    model.load_initial_weights(sess)

    # Loop over all images
    for i, image in enumerate(imgs):
        # Convert image to float32 and resize to (227x227)
        img = cv2.resize(image.astype(np.float32), (227, 227))
        # Subtract the ImageNet mean
        img -= imagenet_mean
        # Reshape as needed to feed into model
        img = img.reshape((1, 227, 227, 3))
        # Run the session and calculate the class probability
        fc8_out, probs = sess.run([score, softmax], feed_dict={x: img, keep_prob: 1})
        print fc8_out.shape
        # Get the class name of the class with the highest probability
        class_name = class_names[np.argmax(probs)]
        print ('Image %s - Class: %s, probability: %.4f'%(i+1, class_name, probs[0,np.argmax(probs)]))

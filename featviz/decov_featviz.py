# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from scipy.misc import imresize
from PIL import Image
import tensorflow as tf


import sys
sys.path.append('../')
from salexnet import AlexNet

from tf_cnnvis import *

base_dir = r'/nfs/home/huanglijie/repo/finetune_alexnet_with_tensorflow'
model_data = os.path.join(base_dir,'log','checkpoints','sel_model_epoch44.ckpt')

current_dir = os.getcwd()

# load test image info
test_data_list = os.path.join(base_dir, 'emoImg', 'test_list.txt')
test_data_list = open(test_data_list, 'r').readlines()
test_data_list = [line.strip().split()[0] for line in test_data_list]

# mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
# model config
t_preprocessed = tf.placeholder(tf.float32, shape=(1, 227, 227, 3))
#t_input = tf.placeholder(tf.float32, shape=(380, 330, 3))
#t_aligned = tf.image.resize_image_with_crop_or_pad(t_input, 380, 380)
#t_resized = tf.image.resize_images(t_aligned, [227, 227])
#t_preprocessed = tf.expand_dims(t_resized-imagenet_mean, 0)
keep_prob = tf.placeholder(tf.float32)
net = AlexNet(t_preprocessed, keep_prob, 4, [])
 
# create tensorflow session
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, model_data)
graph = sess.graph

# get Conv2D layer name
layers = [op.name for op in graph.get_operations() if op.type=='Conv2D']
layers = layers[-2:]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) 
                for name in layers]
print('Number of layers', len(layers))
print('Total number of feature channels:', sum(feature_nums))

# load test image
img = Image.open(test_data_list[0])
img0 = Image.new('RGB', (380, 380))
img0.paste(img, ((380-330)//2, (380-380)//2))
img0 = np.asarray(img0)
img0 = np.expand_dims(imresize(img0, (227, 227))-imagenet_mean, 0)

for layer in layers:
    channel_num = int(graph.get_tensor_by_name(layer+':0').get_shape()[-1])
    for channel in range(channel_num):
        print 'Viz feature of Layer %s, Channel %s'%(layer, channel)

        is_success = deconv_visualization(sess_graph_path=None,
                                          value_feed_dict={t_preprocessed: img0,
                                          #value_feed_dict={t_input: img0,
                                                           keep_prob: 1.},
                                          layers=layer,
                                          path_logdir='EmoNetLog',
                                          path_outdir='EmoNetOut')


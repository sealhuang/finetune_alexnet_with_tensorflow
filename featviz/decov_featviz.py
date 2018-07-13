# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from scipy.ndimage import imread
import tensorflow as tf

import sys
sys.path.append('../')
from salexnet import AlexNet


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
t_input = tf.placeholder(tf.float32, shape=(380, 330, 3))
t_aligned = tf.image.resize_image_with_crop_or_pad(t_input, 380, 380)
t_resized = tf.image.resize_images(t_aligned, [227, 227])
t_preprocessed = tf.expand_dims(t_resized-imagenet_mean, 0)
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

for layer in layers:
    channel_num = int(graph.get_tensor_by_name(layer+':0').get_shape()[-1])
    for channel in range(channel_num):
        print 'Viz feature of Layer %s, Channel %s'%(layer, channel)

        is_success = deconv_visualization(sess_graph_path=None,
                                          value_feed_dict={t_input: img0,
                                                           keep_prob: 1.},
                                          layers=layer,
                                          path_logdir='EmoNetLog',
                                          path_outdir='EmoNetOut')

        # defining the optimization objective
        t_obj = graph.get_tensor_by_name('%s:0'%layer)[:, :, :, channel]
        t_score = tf.reduce_mean(t_obj)
        #t_score = tf.reduce_max(t_obj)
        # behold the power of automatic differentiation!
        t_grad = tf.gradients(t_score, t_input)[0]

        # feature array
        img_feat = np.zeros((380, 330, 3, 800), dtype=np.float32)
        c = 0
        # start with a selected image from testing images
        for img_file in test_data_list:
            img0 = imread(img_file, mode='RGB')
            g, score = sess.run([t_grad, t_score],
                                {t_input: img0, keep_prob: 1.})
            # normalize
            a = g - g.mean(axis=2, keepdims=True)
            a = (a - a.min()) / (a.max() - a.min())
            a[a>0.7] = 1
            img_feat[..., c] = a
            c += 1
            print c
        outfile = os.path.join(current_dir, 'feats',
                               '%s_%s_feats.npy'%(layer, channel))
        np.save(outfile, img_feat)


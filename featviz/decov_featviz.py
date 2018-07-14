# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
test_data_list = os.path.join(current_dir, 'stim4vis.txt')
test_data_list = open(test_data_list, 'r').readlines()
test_data_list = [line.strip().split()[0] for line in test_data_list]

# mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
# model config
t_input = tf.placeholder(tf.float32, shape=(1, 227, 227, 3))
keep_prob = tf.placeholder(tf.float32)
net = AlexNet(t_input, keep_prob, 4, [])
 
# create tensorflow session
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, model_data)
graph = sess.graph

# get Conv2D layer name
layers = ['pool1', 'pool2', 'Conv2D_3', 'Conv2D_4', 'Conv2D_5', 'pool5']

# load test image
for img_num in range(len(test_data_list)):
    basename =os.path.basename(test_data_list[img_num])
    print 'Input %s'%(basename)
    basename = '.'.join(basename.split('.')[0:-1]+['png'])
    img = Image.open(test_data_list[img_num])
    img0 = Image.new('RGB', (380, 380))
    img0.paste(img, ((380-330)//2, (380-380)//2))
    img0 = np.asarray(img0)
    img0 = np.expand_dims(imresize(img0, (227, 227))-imagenet_mean, 0)

    is_success = deconv_visualization(sess_graph_path=sess,
                                      value_feed_dict={t_input: img0,
                                                       keep_prob: 1.},
                                      layers=layers,
                                      path_logdir='EmoNetLog',
                                      path_outdir='EmoNetOut')

    # move images
    sub_dirs = {'pool1': 'pool1', 'pool2': 'pool2', 'pool5': 'pool5',
                'conv2d_3': 'conv3', 'conv2d_4': 'conv4_1',
                'conv2d_5': 'conv4_2'}
    for k in sub_dirs:
        src_img = os.path.join(current_dir, 'EmoNetOut', k, 'deconvolution',
                               'grid_image.png')
        targ_img = os.path.join(current_dir, 'feats', sub_dirs[k], basename)
        cmd_str = ['mv', src_img, targ_img]
        #print ' '.join(cmd_str)
        os.system(' '.join(cmd_str))

    os.system('rm -rf EmoNetLog EmoNetOut model')


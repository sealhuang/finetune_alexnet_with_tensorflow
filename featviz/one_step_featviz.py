# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from scipy.ndimage import imread
import matplotlib.pylab as plt
import tensorflow as tf

import sys
sys.path.append('../')
import salexnet as Alexnet


def savearray(a, filename):
    a = np.uint8(np.clip(a, 0, 1)*255)
    plt.imshow(a, cmap='gray')
    filename = filename.replace('/', '_')
    plt.savefig(filename+'.png')
    plt.close()
    np.save(filename+'.npy', a)

def visstd(a, s=0.1):
    """Normalize the image range for visualization"""
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5


if __name__=='__main__':
    base_dir = r'/nfs/home/huanglijie/repo/finetune_alexnet_woth_tensorflow'
    model_dir = os.path.join(base_dir, 'log', 'checkpoints')
    model_data = os.path.join(model_dir, 'sel_model_epoch10.ckpt')

    # load test image info
    test_data_list = os.path.join(base_dir, 'emoImg', 'test_list.txt')
    test_data_list = open(test_data_list, 'r').readlines()
    test_data_list = [line.strip().split(',')[0] for line in test_data_list]
    test_data_list = test_data_list[0]

    # load the model
    t_input = tf.placeholder(tf.float32, shape=(227, 227, 3))
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
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
            # start with a selected image from testing images
            for img_file in test_data_list:
                img0 = imread(img_file, mode='RGB')
                t_obj = graph.get_tensor_by_name('%s:0'%layer)[:, :, :, channel]
 
                # defining the optimization objective
                t_score = tf.reduce_mean(t_obj)
                #t_score = tf.reduce_max(t_obj)
                # behold the power of automatic differentiation!
                t_grad = tf.gradients(t_score, t_input)[0]

                g, score = sess.run([t_grad, t_score],
                                    {t_input: img, keep_prob: 1.})
                # normalizing the gradient, so the same step size should work
                # for different layers and networks
                g /= g.std() + 1e-8
                print g.shape
                #savearray(visstd(g[0, :, :]), '%s_%s'%(layer, channel))


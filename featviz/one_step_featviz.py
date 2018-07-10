# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
from scipy.ndimage import imread
import matplotlib.pylab as plt
import tensorflow as tf

import sys
sys.path.append('../')
from salexnet import AlexNet


def savearray(a, dir_name, filename):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, mode=0755)
    a = np.uint8(np.clip(a, 0, 1)*255)
    plt.imshow(a)
    plt.savefig(os.path.join(dir_name, filename+'.png'))
    plt.close()
    np.save(os.path.join(dir_name, filename+'.npy'), a)

def visstd(a, s=0.1):
    """Normalize the image range for visualization"""
    #return (a-a.mean())/max(a.std(), 1e-4)*s
    b = a - a.mean(axis=2, keepdims=True)
    b = (b - b.min()) / (b.max() - b.min())
    b[b>0.7] = 1
    return b


if __name__=='__main__':
    base_dir = r'/nfs/home/huanglijie/repo/finetune_alexnet_with_tensorflow'
    model_dir = os.path.join(base_dir, 'log', 'checkpoints')
    model_data = os.path.join(model_dir, 'sel_model_epoch44.ckpt')

    current_dir = os.getcwd()

    # load test image info
    test_data_list = os.path.join(base_dir, 'emoImg', 'test_list.txt')
    test_data_list = open(test_data_list, 'r').readlines()
    test_data_list = [line.strip().split()[0] for line in test_data_list]

    # load the model
    t_input = tf.placeholder(tf.float32, shape=(380, 330, 3))
    t_aligned = tf.image.resize_image_with_crop_or_pad(t_input, 380, 380)
    t_resized = tf.image.resize_images(t_aligned, [227, 227])
    imagenet_mean = 117.0
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
    layers = layers[-4:]
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
                                    {t_input: img0, keep_prob: 1.})
                savearray(visstd(g),
                    os.path.join(current_dir, 'test', '%s_%s'%(layer, channel)),
                    '.'.join(os.path.basename(img_file).split('.')[:-1]))


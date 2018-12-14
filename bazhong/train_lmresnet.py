# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""Script to finetune AffectNet using Tensorflow."""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import tensorflow as tf
from tensorflow.data import Iterator
from datetime import datetime
from random import shuffle as list_shuffle

from resnet18 import ResNet
from imgdatagenerator_lmresnet import ImageDataGenerator


def source_data(beh_file, dist_file):
    """Read sample information, get split train- and test-dataset."""
    # config sample number per class
    all_sample_num = 1500
    train_sample_num = 1350

    # read behavior info
    beh_info = open(beh_file).readlines()
    beh_info = [line.strip().split(',') for line in beh_info]
    vals = [float(line[1]) for line in beh_info]
    # read landmark dist info
    dist = np.load(dist_file)
    dist_mean = dist.mean(axis=0)
    dist = dist - dist_mean
    
    sorted_idx = np.argsort(vals)
    low_part = sorted_idx[0:all_sample_num]
    high_part = sorted_idx[(-1*all_sample_num):]
    low_dist = dist[low_part, :, :]
    high_dist = dist[high_part, :, :]
    rand_low_dist_idx = range(low_dist.shape[0])
    list_shuffle(rand_low_dist_idx)
    low_dist = low_dist[rand_low_dist_idx, :, :]
    rand_high_dist_idx = range(high_dist.shape[0])
    list_shuffle(rand_high_dist_idx)
    high_dist = high_dist[rand_high_dist_idx, :, :]

    train_dist = np.concatenate((low_dist[:train_sample_num, :, :],
                                 high_dist[:train_sample_num, :, :]),
                                axis=0)
    val_dist = np.concatenate((low_dist[train_sample_num:, :, :],
                               high_dist[train_sample_num:, :, :]),
                              axis=0)
    train_labels = [0]*train_sample_num + [1]*train_sample_num
    val_labels = [0]*(all_sample_num-train_sample_num) + \
                 [1]*(all_sample_num-train_sample_num)

    print 'Training set: %s'%(train_dist.shape[0])
    print 'Validation set: %s'%(val_dist.shape[0])

    return train_dist, train_labels, val_dist, val_labels
    
def model_train(train_dist, train_labels, val_dist, val_labels):
    # Learning params
    learning_rate = 0.01
    num_epochs = 500
    batch_size = 30

    # Network params
    dropout_rate = 0.5
    num_classes = 2

    # How often we want to write the tf.summary data to disk
    display_step = 45

    # Path for tf.summary.FileWriter and to store model checkpoints
    current_dir = os.getcwd()
    filewriter_path = os.path.join(current_dir, 'log','lm_cls_tensorboard')
    checkpoint_path = os.path.join(current_dir, 'log','lm_cls_checkpoints')

    #-- Main Part of the finetuning Script.

    # Create parent path if it doesn't exist
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path, mode=0755)

    # Place data loading and preprocessing on the cpu
    with tf.device('/cpu:0'):
        tr_data = ImageDataGenerator(train_dist, train_labels,
                                     mode='training',
                                     batch_size=batch_size,
                                     num_classes=num_classes,
                                     shuffle=True)
        val_data = ImageDataGenerator(val_dist, val_labels,
                                      mode='inference',
                                      batch_size=batch_size,
                                      num_classes=num_classes,
                                      shuffle=False)

        # create an reinitializable iterator given the dataset structure
        iterator = Iterator.from_structure(tr_data.data.output_types,
                                           tr_data.data.output_shapes)
        next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    training_init_op = iterator.make_initializer(tr_data.data)
    validation_init_op = iterator.make_initializer(val_data.data)
    #test_init_op = iterator.make_initializer(test_data.data)

    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [batch_size, 72, 72, 1])
    y = tf.placeholder(tf.float32, [batch_size, num_classes])
    is_train = tf.placeholder(tf.bool, name='is_train')
    #keep_prob = tf.placeholder(tf.float32)

    # Initialize model
    model = ResNet(x, num_classes, [], is_train)

    # Link variable to model output
    score = model.logits

    # List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables()]

    # Op for calculating the loss
    with tf.name_scope("cross_ent"):
        logits = tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                         labels=y)
        loss = tf.reduce_mean(logits)

    # Train op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.name_scope("train"):
        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))
        # Create optimizer and apply gradient descent to the trainable variables
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        #with tf.control_dependencies(update_ops):
        #    train_op = optimizer.apply_gradients(grads_and_vars=gradients)
        #optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, var_list=var_list)

    # Add gradients to summary
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Add the loss to summary
    tf.summary.scalar('cross-ent', loss)

    # Evaluation op: Accuracy of the model
    with tf.name_scope('accuracy'):
        pred_label = tf.argmax(score, 1)
        true_label = tf.argmax(y, 1)
        correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
    val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))
    #test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))
    print 'Train data batches per epoch: %s'%(train_batches_per_epoch)
    print 'Val data batches per epoch: %s'%(val_batches_per_epoch)


    # Start Tensorflow session
    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          filewriter_path))

        # Loop over number of epochs
        for epoch in range(num_epochs):

            print("{} Epoch number: {}".format(datetime.now(), epoch+1))

            # Initialize iterator with the training dataset
            sess.run(training_init_op)
            for step in range(train_batches_per_epoch):
                # get next batch of data
                img_batch, label_batch = sess.run(next_batch)
                # And run the training op
                sess.run(train_op, feed_dict={x: img_batch,
                                              y: label_batch,
                                              is_train: True})
                # Generate summary with the current batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                            y: label_batch,
                                                            is_train: False})
                    writer.add_summary(s, epoch*train_batches_per_epoch + step)

            # Test the model on the entire training set to check over-fitting
            print("{} Start validation".format(datetime.now()))
            val_acc = 0.
            val_count = 0
            sess.run(training_init_op)
            for _ in range(train_batches_per_epoch):
                img_batch, label_batch = sess.run(next_batch)
                acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                    y: label_batch,
                                                    is_train: False})
                val_acc += acc
                val_count += 1
            val_acc /= val_count
            print("{} Validation Accuracy = {:.4f}".format(datetime.now(), val_acc))

            # Test the model on the entire test set
            print("{} Start test".format(datetime.now()))
            sess.run(validation_init_op)
            test_acc = 0.
            test_count = 0
            preds = []
            trues = []
            for _ in range(val_batches_per_epoch):
                img_batch, label_batch = sess.run(next_batch)
                acc, pl, tl = sess.run([accuracy, pred_label, true_label], feed_dict={x: img_batch,
                                                                     y: label_batch,
                                                                     is_train: False})
                test_acc += acc
                test_count += 1
                preds = np.concatenate((preds, pl))
                trues = np.concatenate((trues, tl))
            test_acc /= test_count
            print("{} Test Accuracy = {:.4f}".format(datetime.now(), test_acc))
            print 'Confusion matrix'
            cm = sess.run(tf.confusion_matrix(preds, trues))
            print cm
        

if __name__ == '__main__':
    current_dir = os.getcwd()

    # Path to the textfiles for the dataset
    beh_file = os.path.join(current_dir, 'data', 'norm_landmark_data_list.csv')
    dist_file = os.path.join(current_dir, 'data', 'landmark_dist_mtx.npy')
    train_dist,train_labels,val_dist,val_labels=source_data(beh_file,dist_file)
    model_train(train_dist, train_labels, val_dist, val_labels)


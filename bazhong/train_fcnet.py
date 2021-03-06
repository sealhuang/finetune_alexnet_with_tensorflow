# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow.data import Iterator
from datetime import datetime

from fcnet import FCNet
from datasource import source_landmark_with_age_sampling
from imgdatagenerator import LandmarkDataGenerator


def model_train(train_landmarks, train_labels, val_landmarks, val_labels):
    # learning params
    init_lr = 0.01
    lr_decay = 0.1
    epoch_decay = 20
    num_epochs = 50
    batch_size = 50

    # Network params
    dropout_rate = 0.5
    num_classes = 2
    train_layers = ['fc1', 'fc2', 'fc3', 'fc4', 'bn3']
    
    # How often we want to write the tf.summary data to disk
    display_step = 15

    # Path for tf.summary.FileWriter and to store model checkpoints
    current_dir = os.getcwd()
    filewriter_path = os.path.join(current_dir, 'log', 'fc_tensorboard')
    checkpoint_path = os.path.join(current_dir, 'log', 'fc_checkpoints')

    #-- Main Part of the finetuning Script.

    # Create parent path if it doesn't exist
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path, mode=0755)

    # Place data loading and preprocessing on the cpu
    with tf.device('/cpu:0'):
        tr_data = LandmarkDataGenerator(train_landmarks, train_labels,
                                        mode='training',
                                        batch_size=batch_size,
                                        num_classes=num_classes,
                                        shuffle=True)
        val_data = LandmarkDataGenerator(val_landmarks, val_labels,
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
    x = tf.placeholder(tf.float32, [batch_size, 144])
    y = tf.placeholder(tf.float32, [batch_size, num_classes])
    is_train = tf.placeholder(tf.bool, name='is_train')
    keep_prob = tf.placeholder(tf.float32)
    lr = tf.placeholder(tf.float32)

    # Initialize model
    model = FCNet(x, keep_prob, num_classes, is_train)

    # Link variable to model output
    score = model.logits

    # List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables()
                if v.name.split('/')[0] in train_layers]

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
        #train_op = optimizer.apply_gradients(grads_and_vars=gradients)
        #optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = tf.train.MomentumOptimizer(lr, 0.95, use_nesterov=True)
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

        test_acc_list = []

        # Loop over number of epochs
        for epoch in range(num_epochs):
            print("{} Epoch number: {}".format(datetime.now(), epoch+1))
            # Initialize iterator with the training dataset
            sess.run(training_init_op)

            # calculate learning rate
            current_lr = lr_decay**((epoch+1)/epoch_decay) * init_lr

            for step in range(train_batches_per_epoch):
                # get next batch of data
                lm_batch, label_batch = sess.run(next_batch)
                # And run the training op
                sess.run(train_op, feed_dict={x: lm_batch,
                                              y: label_batch,
                                              keep_prob: dropout_rate,
                                              is_train: True,
                                              lr: current_lr})

                # Generate summary with the current batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: lm_batch,
                                                            y: label_batch,
                                                            keep_prob: 1.,
                                                            is_train: False,
                                                            lr: current_lr})

                    writer.add_summary(s, epoch*train_batches_per_epoch + step)

            # Test the model on the entire training set to check over-fitting
            print("{} Start validation".format(datetime.now()))
            val_acc = 0.
            val_count = 0
            sess.run(training_init_op)
            for _ in range(train_batches_per_epoch):
                lm_batch, label_batch = sess.run(next_batch)
                acc = sess.run(accuracy, feed_dict={x: lm_batch,
                                                    y: label_batch,
                                                    keep_prob: 1.,
                                                    is_train: False,
                                                    lr: current_lr})
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
                lm_batch, label_batch = sess.run(next_batch)
                acc, pl, tl = sess.run([accuracy, pred_label, true_label], feed_dict={x: lm_batch, y: label_batch, keep_prob: 1., is_train: False, lr: current_lr})
                test_acc += acc
                test_count += 1
                preds = np.concatenate((preds, pl))
                trues = np.concatenate((trues, tl))
            test_acc /= test_count
            print("{} Test Accuracy = {:.4f}".format(datetime.now(), test_acc))
            print 'Confusion matrix'
            cm = sess.run(tf.confusion_matrix(preds, trues))
            print cm

            test_acc_list.append(test_acc)

        with open('lm_test_acc.csv', 'a') as f:
            f.write(','.join([str(item) for item in test_acc_list])+'\n')


if __name__ == '__main__':
    current_dir = os.getcwd()

    # path to the textfiles for the dataset
    data_file = os.path.join(current_dir, 'data', 'norm_landmark_data_list.csv')
    train_landmarks, train_labels, val_landmarks, val_labels = source_landmark_with_age_sampling(data_file, rand_val=False, gender=None)
    model_train(train_landmarks, train_labels, val_landmarks, val_labels)


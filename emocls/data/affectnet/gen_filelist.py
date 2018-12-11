# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os

sel_labels = ['Happy', 'Fear', 'Disgust', 'Neutral']

current_dir = os.getcwd()
all_labels = open('label.txt').readlines()
all_labels = [line.strip() for line in all_labels]

train_list_f = open('train_list.txt', 'wb')
for i, l in enumerate(all_labels):
    if l in sel_labels:
        class_dir = os.path.join(current_dir, 'train_class', 'class00%s'%(i+1))
        imglist = os.listdir(class_dir)
        for img in imglist:
            train_list_f.write('%s %s\n'%(os.path.join(class_dir, img), sel_labels.index(l)))
train_list_f.close()

val_list_f = open('val_list.txt', 'wb')
for i, l in enumerate(all_labels):
    if l in sel_labels:
        class_dir = os.path.join(current_dir, 'val_class', 'class00%s'%(i+1))
        imglist = os.listdir(class_dir)
        for img in imglist:
            val_list_f.write('%s %s\n'%(os.path.join(class_dir, img), sel_labels.index(l)))
val_list_f.close()


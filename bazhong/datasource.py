# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
from random import shuffle as list_shuffle


def source_data(data_info_file, img_dir, rand_val=False, gender=None):
    """Read sample information, get split train- and test-dataset."""
    # config sample number per class
    #all_sample_num = 1500
    #train_sample_num = 1350
    all_sample_num = 1000
    train_sample_num = 900

    # read sample info
    all_info = open(data_info_file).readlines()
    all_info.pop(0)
    all_info = [line.strip().split(',') for line in all_info]
    # select specific gender samples
    if gender=='m':
        all_info = [line for line in all_info if int(line[1][16])%2==1]
    elif gender=='f':
        all_info = [line for line in all_info if int(line[1][16])%2==0]
    else:
        pass
    imgs = [os.path.join(img_dir, line[2]) for line in all_info]
    vals = [float(line[3]) for line in all_info]
    ages = []
    for line in all_info:
        birth_year = int(line[1][6:10])
        birth_month = int(line[1][10:12])
        a = 2008 - birth_year + (12-birth_month)*1.0/12
        ages.append(a)
    
    # select samples within specific age range
    vals = [vals[i] for i in range(len(ages)) if ages[i]>=1.5]
    imgs = [imgs[i] for i in range(len(ages)) if ages[i]>=1.5]
    ages = [item for item in ages if item>=1.5]

    # sort the IQs, and split dataset into high and low parts
    sorted_idx = np.argsort(vals)
    low_part = sorted_idx[0:all_sample_num]
    high_part = sorted_idx[(-1*all_sample_num):]
    low_imgs = [imgs[i] for i in low_part]
    high_imgs = [imgs[i] for i in high_part]
    low_ages = [ages[i] for i in low_part]
    high_ages = [ages[i] for i in high_part]
    # shuffle the sample parts
    rand_low_idx = range(len(low_imgs))
    list_shuffle(rand_low_idx)
    low_imgs = [low_imgs[i] for i in rand_low_idx]
    low_ages = [low_ages[i] for i in rand_low_idx]
    rand_high_idx = range(len(high_imgs))
    list_shuffle(rand_high_idx)
    high_imgs = [high_imgs[i] for i in rand_high_idx]
    high_ages = [high_ages[i] for i in rand_high_idx]
    
    train_imgs = low_imgs[:train_sample_num] + high_imgs[:train_sample_num]
    train_ages = low_ages[:train_sample_num] + high_ages[:train_sample_num]
    val_imgs = low_imgs[train_sample_num:] + high_imgs[train_sample_num:]
    val_ages = low_ages[train_sample_num:] + high_ages[train_sample_num:]
    train_labels = [0]*train_sample_num + [1]*train_sample_num
    val_labels = [0]*(all_sample_num-train_sample_num) + \
                 [1]*(all_sample_num-train_sample_num)
    if rand_val:
        list_shuffle(val_labels)

    return train_imgs, train_labels, val_imgs, val_labels
    
def source_data_with_age(data_info_file, img_dir, rand_val=False, gender=None):
    """Read sample information, get split train- and test-dataset."""
    # config sample number per class
    #all_sample_num = 1500
    #train_sample_num = 1350
    all_sample_num = 1000
    train_sample_num = 900

    # read sample info
    all_info = open(data_info_file).readlines()
    all_info.pop(0)
    all_info = [line.strip().split(',') for line in all_info]
    # select specific gender samples
    if gender=='m':
        all_info = [line for line in all_info if int(line[1][16])%2==1]
    elif gender=='f':
        all_info = [line for line in all_info if int(line[1][16])%2==0]
    else:
        pass
    imgs = [os.path.join(img_dir, line[2]) for line in all_info]
    vals = [float(line[3]) for line in all_info]
    ages = []
    for line in all_info:
        birth_year = int(line[1][6:10])
        birth_month = int(line[1][10:12])
        a = 2008 - birth_year + (12-birth_month)*1.0/12
        ages.append(a)
    
    # select samples within specific age range
    vals = [vals[i] for i in range(len(ages)) if ages[i]>=1.5]
    imgs = [imgs[i] for i in range(len(ages)) if ages[i]>=1.5]
    ages = [item for item in ages if item>=1.5]

    # sort the IQs, and split dataset into high and low parts
    sorted_idx = np.argsort(vals)
    low_part = sorted_idx[0:all_sample_num]
    high_part = sorted_idx[(-1*all_sample_num):]
    low_imgs = [imgs[i] for i in low_part]
    high_imgs = [imgs[i] for i in high_part]
    low_ages = [ages[i] for i in low_part]
    high_ages = [ages[i] for i in high_part]
    # shuffle the sample parts
    rand_low_idx = range(len(low_imgs))
    list_shuffle(rand_low_idx)
    low_imgs = [low_imgs[i] for i in rand_low_idx]
    low_ages = [low_ages[i] for i in rand_low_idx]
    rand_high_idx = range(len(high_imgs))
    list_shuffle(rand_high_idx)
    high_imgs = [high_imgs[i] for i in rand_high_idx]
    high_ages = [high_ages[i] for i in rand_high_idx]
    
    train_imgs = low_imgs[:train_sample_num] + high_imgs[:train_sample_num]
    train_ages = low_ages[:train_sample_num] + high_ages[:train_sample_num]
    val_imgs = low_imgs[train_sample_num:] + high_imgs[train_sample_num:]
    val_ages = low_ages[train_sample_num:] + high_ages[train_sample_num:]
    train_labels = [0]*train_sample_num + [1]*train_sample_num
    val_labels = [0]*(all_sample_num-train_sample_num) + \
                 [1]*(all_sample_num-train_sample_num)
    if rand_val:
        list_shuffle(val_labels)

    return train_imgs, train_ages, train_labels, val_imgs, val_ages, val_labels
 

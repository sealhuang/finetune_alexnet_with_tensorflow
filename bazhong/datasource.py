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
    # select samples of specific gender
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
    imgs = [imgs[i] for i in range(len(ages)) if ages[i]>=1.5]
    vals = [vals[i] for i in range(len(ages)) if ages[i]>=1.5]
    ages = [item for item in ages if item>=1.5]
    print('%s samples collected'%(len(imgs)))

    # sort the IQs, and split dataset into high and low parts
    sorted_idx = np.argsort(vals)
    low_part = sorted_idx[0:all_sample_num]
    high_part = sorted_idx[(-1*all_sample_num):]
    low_imgs = [imgs[i] for i in low_part]
    high_imgs = [imgs[i] for i in high_part]
    #low_ages = [ages[i] for i in low_part]
    #high_ages = [ages[i] for i in high_part]
    # shuffle the sample parts
    rand_low_idx = range(len(low_imgs))
    list_shuffle(rand_low_idx)
    low_imgs = [low_imgs[i] for i in rand_low_idx]
    #low_ages = [low_ages[i] for i in rand_low_idx]
    rand_high_idx = range(len(high_imgs))
    list_shuffle(rand_high_idx)
    high_imgs = [high_imgs[i] for i in rand_high_idx]
    #high_ages = [high_ages[i] for i in rand_high_idx]
    
    train_imgs = low_imgs[:train_sample_num] + high_imgs[:train_sample_num]
    #train_ages = low_ages[:train_sample_num] + high_ages[:train_sample_num]
    val_imgs = low_imgs[train_sample_num:] + high_imgs[train_sample_num:]
    #val_ages = low_ages[train_sample_num:] + high_ages[train_sample_num:]
    train_labels = [0]*train_sample_num + [1]*train_sample_num
    val_labels = [0]*(all_sample_num-train_sample_num) + \
                 [1]*(all_sample_num-train_sample_num)
    if rand_val:
        list_shuffle(val_labels)

    print('Training samples %s'%(len(train_imgs)))
    print('Validation samples %s'%(len(val_imgs)))

    return train_imgs, train_labels, val_imgs, val_labels
    
def source_data_expand(data_info_file, img_dir, rand_val=False, gender=None):
    """Read sample information, get split train- and test-dataset."""
    # config sample number per class
    train_sample_num = 1000
    val_sample_num = 1000

    # read sample info
    all_info = open(data_info_file).readlines()
    all_info.pop(0)
    all_info = [line.strip().split(',') for line in all_info]
    # select samples of specific gender
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
    
    # select training samples within specific age range
    timgs = [imgs[i] for i in range(len(ages)) if ages[i]>=1.5]
    tvals = [vals[i] for i in range(len(ages)) if ages[i]>=1.5]
    print('%s training samples collected'%(len(timgs)))

    # select validation samples within specific age range
    vimgs = [imgs[i] for i in range(len(ages)) if (ages[i]>=1.0) and (ages[i]<1.5)]
    vvals = [vals[i] for i in range(len(ages)) if (ages[i]>=1.0) and (ages[i]<1.5)]
    print('%s validation samples collected'%(len(vimgs)))

    # sort the IQs, and split dataset into high and low parts
    train_sorted_idx = np.argsort(tvals)
    train_low_part = train_sorted_idx[0:train_sample_num]
    train_high_part = train_sorted_idx[(-1*train_sample_num):]
    train_idx = train_low_part.tolist() + train_high_part.tolist()
    train_imgs = [timgs[i] for i in train_idx]
    train_labels = [0]*train_sample_num + [1]*train_sample_num
    
    val_sorted_idx = np.argsort(vvals)
    val_low_part = val_sorted_idx[0:val_sample_num]
    val_high_part = val_sorted_idx[(-1*val_sample_num):]
    val_idx = val_low_part.tolist() + val_high_part.tolist()
    val_imgs = [vimgs[i] for i in val_idx]
    val_labels = [0]*val_sample_num + [1]*val_sample_num
    
    
    if rand_val:
        list_shuffle(val_labels)

    print('Training samples %s'%(len(train_imgs)))
    print('Validation samples %s'%(len(val_imgs)))

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
 
def source_data_with_age_sampling(data_info_file, img_dir, sample_num,
                                  small_sample_num, val_set_idx,
                                  rand_val=False, gender=None):
    """Read sample information, get split train- and test-dataset."""
    # config sample number per class
    #sample_num = 100
    #small_sample_num = 75

    # read sample info
    all_info = open(data_info_file).readlines()
    all_info.pop(0)
    all_info = [line.strip().split(',') for line in all_info]
    # select specific gender samples
    if gender=='m':
        all_info = [line for line in all_info if int(line[0][16])%2==1]
    elif gender=='f':
        all_info = [line for line in all_info if int(line[0][16])%2==0]
    else:
        pass

    # remove samples with glasses
    all_info = [line for line in all_info if not int(line[1])]

    # remove 2 2006-students
    all_info = [line for line in all_info if not int(line[0][6:10])==2006]

    print('%s samples selected'%(len(all_info)))

    imgs = []
    for line in all_info:
        tmp = []
        for item in line[6:]:
            tmp.append(os.path.join(img_dir, item))
        imgs.append(tmp)
    # index 1: graphic reasoning, 2: expanding graph
    vals = [float(line[2]) for line in all_info]
    ages = []
    for line in all_info:
        birth_year = int(line[0][6:10])
        birth_month = int(line[0][10:12])
        a = 2008 - birth_year + (12-birth_month)*1.0/12
        ages.append(a)
   
    # select samples within each age group
    img_list = []
    label_list = []
    unique_ages = np.unique(ages)
    for a in unique_ages:
        tmp_imgs = [imgs[i] for i in range(len(ages)) if ages[i]==a]
        tmp_vals = [vals[i] for i in range(len(ages)) if ages[i]==a]
        if len(tmp_imgs)<(2*sample_num):
            snum = small_sample_num
        else:
            snum = sample_num
        # select top- and bottom- part of samples
        sorted_idx = np.argsort(tmp_vals)
        low_part = sorted_idx[:snum]
        high_part = sorted_idx[-snum:]
        sel_idx = np.concatenate((low_part, high_part))
        tmp_imgs = [tmp_imgs[i] for i in sel_idx]
        tmp_labels = [0]*snum + [1]*snum
        # shuffle the sample parts
        rand_idx = list(range(len(tmp_imgs)))
        list_shuffle(rand_idx)
        tmp_imgs = [tmp_imgs[i] for i in rand_idx]
        tmp_labels = [tmp_labels[i] for i in rand_idx]
        img_list.append(tmp_imgs)
        label_list.append(tmp_labels)

    # select two subsets of the age groups as validation dataset
    val_set_ids = []
    #for i in range(len(img_list)):
    #    for j in range(i+1, len(img_list)):
    #        val_set_ids.append([i, j])
    #val_set_ids = val_set_ids[val_set_idx]
    #set_ids = range(len(img_list))
    #train_set_ids = [item for item in set_ids if not item in val_set_ids]
    #print 'Age groups for validation: %s and %s'%(unique_ages[val_set_ids[0]],
    #                                              unique_ages[val_set_ids[1]])
    for i in range(len(img_list)):
        val_set_ids.append([i])
    val_set_ids = val_set_ids[val_set_idx]
    set_ids = range(len(img_list))
    train_set_ids = [item for item in set_ids if not item in val_set_ids]
    print('Age groups for validation: %s'%(unique_ages[val_set_ids[0]]))


    val_imgs = [line for i in val_set_ids
                     for line in img_list[i]]
    val_labels = [item for i in val_set_ids
                       for item in label_list[i]]
    train_imgs = [line for i in train_set_ids
                       for line in img_list[i]]
    train_labels = [item for i in train_set_ids
                         for item in label_list[i]]

    # make a full size image array
    train_mlen = max([len(line) for line in train_imgs])
    train_imgs = [line+['null']*(train_mlen-len(line)) for line in train_imgs]
    val_mlen = max([len(line) for line in val_imgs])
    val_imgs = [line+['null']*(val_mlen-len(line)) for line in val_imgs]

    if rand_val:
        list_shuffle(val_labels)

    print('Training samples %s'%(len(train_imgs)))
    print('Validation samples %s'%(len(val_imgs)))

    return train_imgs, train_labels, val_imgs, val_labels
 
def source_landmark_with_age_sampling(data_file, rand_val=False, gender=None):
    """Read sample information, get split train- and test-dataset."""
    # config sample number per class
    sample_num = 100
    small_sample_num = 75

    # read sample info
    all_info = open(data_file).readlines()
    all_info = [line.strip().split(',') for line in all_info]
    # select specific gender samples
    if gender=='m':
        all_info = [line for line in all_info
                        if int(line[0].split('_')[0][16])%2==1]
    elif gender=='f':
        all_info = [line for line in all_info
                        if int(line[0].split('_')[0][16])%2==0]
    else:
        pass
    # get landmarks and IQs
    landmarks = [[float(line[2+i]) for i in range(144)] for line in all_info]
    vals = [float(line[1]) for line in all_info]
    ages = []
    for line in all_info:
        birth_year = int(line[0].split('_')[0][6:10])
        birth_month = int(line[0].split('_')[0][10:12])
        a = 2008 - birth_year + (12-birth_month)*1.0/12
        ages.append(a)
   
    # select samples within each age group
    landmark_list = []
    label_list = []
    for a in np.unique(ages):
        tmp_landmarks = [landmarks[i] for i in range(len(ages)) if ages[i]==a]
        tmp_vals = [vals[i] for i in range(len(ages)) if ages[i]==a]
        if len(tmp_landmarks)<200:
            snum = small_sample_num
        else:
            snum = sample_num
        # select top- and bottom- part of samples
        sorted_idx = np.argsort(tmp_vals)
        low_part = sorted_idx[:snum]
        high_part = sorted_idx[-snum:]
        sel_idx = np.concatenate((low_part, high_part))
        tmp_landmarks = [tmp_landmarks[i] for i in sel_idx]
        tmp_labels = [0]*snum + [1]*snum
        # shuffle the sample parts
        rand_idx = range(len(tmp_landmarks))
        list_shuffle(rand_idx)
        tmp_landmarks = [tmp_landmarks[i] for i in rand_idx]
        tmp_labels = [tmp_labels[i] for i in rand_idx]
        landmark_list.append(tmp_landmarks)
        label_list.append(tmp_labels)

    # select two subsets of the age groups as validation dataset
    group_idx = range(len(landmark_list))
    list_shuffle(group_idx)
    landmark_list = [landmark_list[i] for i in group_idx]
    label_list = [label_list[i] for i in group_idx]
    val_landmarks = []
    val_labels = []
    val_landmarks = [item for line in landmark_list[:2] for item in line]
    val_labels = [item for line in label_list[:2] for item in line]
    train_landmarks = [item for line in landmark_list[2:] for item in line]
    train_labels = [item for line in label_list[2:] for item in line]

    if rand_val:
        list_shuffle(val_labels)

    print('Training samples %s'%(len(train_landmarks)))
    print('Validation samples %s'%(len(val_landmarks)))

    return train_landmarks, train_labels, val_landmarks, val_labels
 
def source_mbti_data(data_info_file, img_dir, rand_val=False, gender=None):
    """Read sample information, get split train- and test-dataset."""
    # config sample selection criteria for each class
    # for E-I factor, <8 : 2971 subjects, >13: 3070 subjects
    low_thresh = 8
    high_thresh = 13
    # select 200 subjects from each class for validation
    val_sample_num = 200

    # read sample info
    all_info = open(data_info_file).readlines()
    all_info.pop(0)
    all_info = [line.strip().split(',') for line in all_info]
    # select samples of specific gender
    if gender=='m':
        all_info = [line for line in all_info if line[1]=='male']
    elif gender=='f':
        all_info = [line for line in all_info if line[1]=='female']
    else:
        pass
    imgs = [os.path.join(img_dir, line[9]) for line in all_info]
    vals = [float(line[5]) for line in all_info]

    # select samples based on specific criteria
    assert len(imgs)==len(vals)
    high_imgs = []
    low_imgs = []
    for i in range(len(vals)):
        if vals[i]<low_thresh:
            low_imgs.append(imgs[i])
        elif vals[i]>high_thresh:
            high_imgs.append(imgs[i])

    # shuffle the samples and split training/validation dataset
    list_shuffle(low_imgs)
    list_shuffle(high_imgs)
    train_imgs = low_imgs[:(-1*val_sample_num)] + high_imgs[:(-1*val_sample_num)]
    val_imgs = low_imgs[(-1*val_sample_num):] + high_imgs[(-1*val_sample_num):]
    train_labels = [0]*(len(low_imgs)-val_sample_num) + [1]*(len(high_imgs)-val_sample_num)
    val_labels = [0]*val_sample_num + [1]*val_sample_num
    if rand_val:
        list_shuffle(val_labels)

    print('Training samples %s - %s'%(len(train_imgs), len(train_labels)))
    print('Validation samples %s - %s'%(len(val_imgs), len(val_labels)))

    return train_imgs, train_labels, val_imgs, val_labels
 

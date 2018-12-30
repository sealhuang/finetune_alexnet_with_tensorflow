# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np

def load_landmark_data(data_file='norm_landmark_data_list.csv'):
    data = open(data_file).readlines()
    data = [line.strip().split(',') for line in data]
    ids = [line[0].split('_')[0] for line in data]
    iqs = [float(line[1]) for line in data]
    gender = []
    for item in ids:
        if int(item[16])%2:
            gender.append(1)
        else:
            gender.append(2)
    birth_year = [int(item[6:10]) for item in ids]
    birth_month = [int(item[10:12]) for item in ids]
    ages = []
    for i in range(len(ids)):
        ages.append(2008-birth_year[i]+(12-birth_month[i])*1.0/12)

    data = {'id': ids,
            'iq': np.array(iqs),
            'gender': np.array(gender),
            'age': np.array(ages)}
    
    return data

def load_data(data_file='data_list.csv'):
    data = open(data_file).readlines()
    data.pop(0)
    data = [line.strip().split(',') for line in data]
    ids = [line[1] for line in data]
    iqs = [float(line[3]) for line in data]
    gender = []
    for item in ids:
        if int(item[16])%2:
            gender.append(1)
        else:
            gender.append(2)
    birth_year = [int(item[6:10]) for item in ids]
    birth_month = [int(item[10:12]) for item in ids]
    ages = []
    for i in range(len(ids)):
        ages.append(2008-birth_year[i]+(12-birth_month[i])*1.0/12)

    data = {'id': ids,
            'iq': np.array(iqs),
            'gender': np.array(gender),
            'age': np.array(ages)}
    
    return data

def age_sampling(data):
    iq_list = []
    age_list = []
    for a in np.unique(data['age']):
        tmp_iq = data['iq'][data['age']==a]
        tmp_age = data['age'][data['age']==a]
        if len(tmp_iq)<200:
            snum = 80
        else:
            snum = 100
        tmp_iq.sort()
        iq_list.append(np.concatenate((tmp_iq[:snum], tmp_iq[-snum:])))
        age_list.append(np.concatenate((tmp_age[:snum], tmp_age[-snum:])))
    return iq_list, age_list


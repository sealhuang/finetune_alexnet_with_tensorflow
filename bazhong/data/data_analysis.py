# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def zone_dist(data):
    """Get the distribution across zones of BJ."""
    zone_info = {'110101': 'DongCheng', '110102': 'XiCheng',
                 '110103': 'ChongWen', '110104': 'XuanWu',
                 '110105': 'ChaoYang', '110106': 'FengTai',
                 '110107': 'ShiJingShan', '110108': 'HaiDian',
                 '110109': 'MenTouGou', '110111': 'FangShan',
                 '110112': 'TongZhou', '110113': 'ShunYi',
                 '110114': 'ChangPing', '110115': 'DaXing',
                 '110116': 'HuaiRou', '110117': 'PingGu',
                 '110228': 'MiYun', '110229': 'YanQing'}
    
    iqs_of_zone = {}
    for i in range(len(data['id'])):
        id_str = data['id'][i]
        if id_str[:6] in zone_info:
            if not id_str[:6] in iqs_of_zone:
                iqs_of_zone[id_str[:6]] = [data['iq'][i]]
            else:
                iqs_of_zone[id_str[:6]].append(data['iq'][i])
   
    f = plt.figure(figsize=(12, 7))
    # plot distribution of student number
    sorted_key = sorted(iqs_of_zone)
    student_num = [len(iqs_of_zone[key]) for key in sorted_key]
    ax1 = f.add_subplot(211)
    ax1.bar(range(len(sorted_key)), student_num, align='center')
    ax1.set_xlim(-0.6, 17.5)
    plt.xticks([], [])
    #plt.xticks(range(len(sorted_key)), [zone_info[key] for key in sorted_key],
    #           rotation=30)
    plt.title('Distribution of student number')
    # plot iq distribution for each student
    df = pd.DataFrame()
    for key in sorted_key:
        x = pd.DataFrame({key: iqs_of_zone[key]})
        df = pd.concat([df, x], axis=1, ignore_index=True)
    ax2 = f.add_subplot(212)
    sns.boxplot(data=df, whis='range', ax=ax2)
    sns.swarmplot(data=df, size=2, color='.3', linewidth=0, ax=ax2)
    plt.xticks(range(len(sorted_key)), [zone_info[key] for key in sorted_key],
               rotation=30)
    plt.ylabel('IQ')
    plt.title('Distribution of IQs')
    plt.savefig('iq_zone_dist.png')
    
    return zone_info, iqs_of_zone

def get_img_stats(img_dir, data_file='data_list.csv'):
    """Get mean and std of images."""
    data = open(data_file).readlines()
    data.pop(0)
    data = [line.strip().split(',') for line in data]
    imgs = [os.path.join(img_dir, line[2]) for line in data]

    img_vals = []
    for img in imgs:
        x = plt.imread(img)
        x = x.reshape(-1, 3)
        mx = x.mean(axis=0)
        img_vals.append(mx)
    img_vals = np.array(img_vals)
    print img_vals.shape
    m = img_vals.mean(axis=0)
    s = img_vals.std(axis=0)
    return m, s


# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import matplotlib.pylab as plt


def savearray(a, dir_name, filename):
    a = np.uint8(np.clip(a, 0, 1)*255)
    plt.imshow(a)
    plt.savefig(os.path.join(dir_name, filename+'.png'))
    plt.close()

def visstd(a):
    """Normalize the image range for visualization"""
    return (a - a.min()) / (a.max() - a.min())


if __name__=='__main__':
    base_dir = os.getcwd()
    feat_dir = os.path.join(base_dir, 'feats')

    # load stimuli info
    stimuli_group_file = os.path.join(base_dir, 'stimuli_group.csv')
    stimuli_group = open(stimuli_group_file).readlines()
    stimuli_group.pop(0)
    stimuli_group = [line.strip().split(',') for line in stimuli_group]
    # get stimuli index
    emo_type = ['Happy', 'Fear', 'Disgust', 'Neutral']
    gender = ['Male', 'Female']
    indexs = {}
    for emo in emo_type:
        for g in gender:
            l = []
            for i in range(len(stimuli_group)):
                if stimuli_group[i][1]==emo and stimuli_group[i][2]==g:
                    l.append(int(i))
            indexs['%s_%s'%(emo, g)] = l

    layer_list = os.listdir(feat_dir)
    for layer in layer_list:
        layer_file = os.path.join(feat_dir, layer)
        feats = np.load(layer_file)
        basename = layer.split('.')[0]
        # mean feature across all images
        mean_feats = feats.mean(axis=3)
        savearray(visstd(mean_feats), feat_dir, basename+'_mean')
        ## mean feature across each emotion type and gender
        #for cond in indexs:
        #    cond_feats = feats[..., indexs[cond]]
        #    mean_feats = cond_feats.mean(axis=3)
        #    savearray(visstd(mean_feats), feat_dir, basename+'_mean_'+cond)


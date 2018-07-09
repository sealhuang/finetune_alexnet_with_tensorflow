# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os

current_dir = os.getcwd()
stim_info = open('stim_info.csv').readlines()
stim_info.pop(0)
stim_info = [line.strip().split(',') for line in stim_info]

test_list_f = open('test_list.txt', 'wb')
for line in stim_info:
    img_file = os.path.join(current_dir, 'imgs', line[0])
    test_list_f.write('%s %s\n'%(img_file, str(int(line[1])-1)))
test_list_f.close()



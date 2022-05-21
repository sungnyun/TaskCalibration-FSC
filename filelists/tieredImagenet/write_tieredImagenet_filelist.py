'''
1. download tieredImagenet dataset, following https://github.com/yaoyao-liu/tiered-imagenet-tools
2. place tiered_imagenet folder (that has train, val, test subdirs) to data/tieredImagenet/images
3. run this code
'''
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import re

cwd = os.getcwd() 
data_path = join(cwd,'images')
savedir = './'
dataset_list = ['base', 'val', 'novel']

cl = -1
folderlist = []

datasetmap = {'base':'train','val':'val','novel':'test'};
filelists = {'base':{},'val':{},'novel':{}}
filelists_flat = {'base':[],'val':[],'novel':[]}
labellists_flat = {'base':[],'val':[],'novel':[]}

for dataset in dataset_list:
    for label in listdir(join(data_path, datasetmap[dataset])):
        label_path = join(data_path, datasetmap[dataset], label)
        label_imgs = [f for f in listdir(label_path) if isfile(join(label_path, f))]

        folderlist.append(label)
        filelists[dataset][label] = []

        for fid in label_imgs:
            fname = join(label_path, fid)
            filelists[dataset][label].append(fname)

    for key, filelist in filelists[dataset].items():
        cl += 1
        # random.shuffle(filelist)
        filelists_flat[dataset] += filelist
        labellists_flat[dataset] += np.repeat(cl, len(filelist)).tolist() 

for dataset in dataset_list:
    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folderlist])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in filelists_flat[dataset]])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in labellists_flat[dataset]])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import deepdish as dd
import pickle as pkl
import numpy as np
import skimage
import skimage.io

import scipy.misc as m
import skimage.transform
from utils import recursive_glob

valid_colors = [[128,  64, 128],
                [244,  35, 232],
                [ 70,  70,  70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170,  30],
                [220, 220,   0],
                [107, 142,  35],
                [152, 251, 152],
                [ 70, 130, 180],
                [220,  20,  60],
                [255,   0,   0],
                [  0,   0, 142],
                [  0,   0,  70],
                [  0,  60, 100],
                [  0,  80, 100],
                [  0,   0, 230],
                [119,  11,  32]]
label_valid_colours = dict(zip(range(19), valid_colors))

void_colors = [[  0,   0,   0],
               [111,  74,   0],
               [ 81,   0,  81],
               [250, 170, 160],
               [230, 150, 140],
               [180, 165, 180],
               [150, 100, 100],
               [150, 120,  90],
               [153, 153, 153],
               [  0,   0,  90],
               [  0,   0, 110]]
label_void_colours = dict(zip(range(11), void_colors))

root = '/home/wilson/dataset/GTA5/'
mean = np.array([73.15835921, 82.90891754, 72.39239876])     
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',\
               'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',\
               'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
               'motorcycle', 'bicycle']

ignore_index = 250
valid_class_map = dict(zip(range(19), valid_colors))
void_class_map  = dict(zip(range(11), void_colors))
class_map = dict(zip(valid_classes, range(19)))
img_size=[512, 1024]
n_class = 19

def transform(img, lbl):
    # img = img[:, :, ::-1]
    img = img.astype(np.float32)
    img -= mean
    img = m.imresize(img, (img_size[0], img_size[1]))
    # Resize scales images from 0 to 255, thus we need
    # to divide by 255.0
    # img = img.astype(float) / 255.0
    # NHWC -> NCWH
    # img = img.transpose(2, 0, 1)

    classes = np.unique(lbl)
    lbl = lbl.astype(float)
    lbl = m.imresize(lbl, (img_size[0], img_size[1]), 'nearest', mode='F')
    lbl = lbl.astype(int)

    if not np.all(np.unique(lbl[lbl!=ignore_index]) < n_class):
        print('after det', classes,  np.unique(lbl))
        raise ValueError("Segmentation map contained invalid class values")
    return img, lbl

X = {}
Y = {}
for split in ['train', 'val']:
    images_base = os.path.join(root, split, 'images')
    annotations_base = os.path.join(root, split, 'labels')
    files = recursive_glob(rootdir=images_base, suffix='.png')
    print("Found %d images" % (len(files)))
    X[split] = np.zeros([len(files), img_size[0], img_size[1], 3], np.uint8)
    Y[split] = np.zeros([len(files), img_size[0], img_size[1]], np.uint8)
    # X[split] = np.zeros([len(files), img_size[0], img_size[1], 3], np.uint8)
    # Y[split] = np.zeros([len(files), img_size[0], img_size[1]], np.uint8)

    def encode_segmap(mask):
        #Put all void classes to zero
        label = np.zeros(mask.shape[:2])
        
        for cls in valid_class_map:
            r_index = (mask[:, :, 0] == valid_class_map[cls][0])
            g_index = (mask[:, :, 1] == valid_class_map[cls][1])
            b_index = (mask[:, :, 2] == valid_class_map[cls][2])
            label[np.logical_and(np.logical_and(r_index, g_index), b_index)] = cls
        for cls in void_class_map:
            r_index = (mask[:, :, 0] == void_class_map[cls][0])
            g_index = (mask[:, :, 1] == void_class_map[cls][1])
            b_index = (mask[:, :, 2] == void_class_map[cls][2])
            label[np.logical_and(np.logical_and(r_index, g_index), b_index)] = ignore_index
        return np.int16(label)
    
    for idx in range(len(files)):
        img_path = files[idx].rstrip()
        lbl_path = os.path.join(annotations_base,
                                os.path.basename(img_path))
        print (img_path)
        print (lbl_path)
        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = encode_segmap(np.array(lbl, dtype=np.uint8))
        
        img, lbl = transform(img, lbl)

        X[split][idx] = img
        Y[split][idx] = lbl
# Save dataset as pickle
'''
with open('gta_data.pkl', 'wb') as f:
    pkl.dump({ 'train': {'images': X['train'], 'labels': Y['train']},
               'val'  : {'images': X['val']  , 'labels': Y['val']}
             }, f, pkl.HIGHEST_PROTOCOL)
'''
d = { 'train': {'images': X['train'], 'labels': Y['train']},
      'val'  : {'images': X['val']  , 'labels': Y['val']},
    }
dd.io.save('GTAData.h5', d)

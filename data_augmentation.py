# -*- coding: utf-8 -*-

#==============================================================================
# 20170531 data augmentation file enlarges and shuffles dataset.npz
#==============================================================================

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

amp_factor = 1.1
shr_factor = 0.9
left_mov_factor = 3
right_mov_factor = 3

data = np.load('data/dataset.npz')['arr_0']
labels = np.load('data/labels.npz')['arr_0']

# data augmentation factor
labels_arr, labels_counts = np.unique(labels, return_counts=True)
labels_counts_aug = np.round(np.sqrt(np.sqrt(labels_counts)*10)*10).astype(np.uint8)
labels_counts_factor = labels_counts_aug/labels_counts
labels_counts_aug2 = labels_counts*labels_counts_factor
print(labels_counts)
print(labels_counts_factor)
print(labels_counts_aug2)

dataset = {}
for i in labels_arr:
    dataset[i] = data[labels == i]

#print(dataset.keys())
#for k in dataset.keys():
#    print(k)
#    print(dataset[k].shape)
    
for k in dataset.keys():
    dataset[k] = np.repeat(dataset[k], 
           labels_counts_factor[k.astype(np.uint8)], 
           axis=0)
    
#plt.clf()
#for i in range(8):
#    plt.subplot(8, 1, i+1)
#    plt.plot(dataset[4][0, i])
#plt.show()
    
for k in dataset.keys():
    dataset[k][0, :8] *= amp_factor
    dataset[k][1, :8] *= shr_factor
    dataset[k][2, :8, :1000-left_mov_factor] = dataset[k][2, :8, left_mov_factor:]
    dataset[k][2, :8, 1000-left_mov_factor:] = 0.0
    dataset[k][3, :8, right_mov_factor:] = dataset[k][3, :8, :1000-right_mov_factor]
    dataset[k][3, :8, :right_mov_factor] = 0.0
    dataset[k][4, :8] *= amp_factor
    dataset[k][4, :8, :1000-left_mov_factor] = dataset[k][4, :8, left_mov_factor:]
    dataset[k][4, :8, 1000-left_mov_factor:] = 0.0
    dataset[k][5, :8] *= amp_factor
    dataset[k][5, :8, right_mov_factor:] = dataset[k][5, :8, :1000-right_mov_factor]
    dataset[k][5, :8, :right_mov_factor] = 0.0
    dataset[k][6, :8] *= shr_factor
    dataset[k][6, :8, :1000-left_mov_factor] = dataset[k][6, :8, left_mov_factor:]
    dataset[k][6, :8, 1000-left_mov_factor:] = 0.0
    dataset[k][7, :8] *= shr_factor
    dataset[k][7, :8, right_mov_factor:] = dataset[k][7, :8, :1000-right_mov_factor]
    dataset[k][7, :8, :right_mov_factor] = 0.0
    
plt.clf()
for i in range(8):
    for j in range(8):
        plt.subplot(8, 1, j+1)
        plt.plot(dataset[15][i, j])
plt.savefig('data_aug_eg.png', dpi=300)


#data_aug = [dataset[k] for k in dataset.keys()]
#label_aug = [[k]*dataset[k].shape[0] for k in dataset.keys()]
#data_aug = np.concatenate(data_aug, axis=0)
#label_aug = np.concatenate(label_aug, axis=0)
#print(data_aug.shape)
#print(label_aug.shape)
#
#np.savez_compressed('data_ext/data_ext.npz', data_aug)
#np.savez_compressed('data_ext/labels_ext.npz', label_aug)
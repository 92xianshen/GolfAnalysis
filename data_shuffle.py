# -*- coding: utf-8 -*-

#==============================================================================
# data shuffle
#==============================================================================

from __future__ import print_function

import numpy as np

data_ext = np.load('data_ext/data_ext.npz')['arr_0']
labels_ext = np.load('data_ext/labels_ext.npz')['arr_0']

indices = np.arange(data_ext.shape[0])
np.random.shuffle(indices)
data_ext_shuffle = data_ext[indices]
labels_ext_shuffle = labels_ext[indices]

np.savez_compressed('data_ext_shuffle/data_ext_shuffle.npz', data_ext_shuffle)
np.savez_compressed('data_ext_shuffle/labels_ext_shuffle.npz', labels_ext_shuffle)
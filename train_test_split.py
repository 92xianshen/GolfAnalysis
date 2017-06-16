# -*- coding: utf-8 -*-

#==============================================================================
# split dataset into training set and test test
#==============================================================================

from __future__ import print_function

import numpy as np
from sklearn.model_selection import train_test_split

data = np.load('data_ext_shuffle/data_ext_shuffle.npz')['arr_0']
labels = np.load('data_ext_shuffle/labels_ext_shuffle.npz')['arr_0']

train_fake = np.arange(data.shape[0]).reshape((data.shape[0], 1))

X_train_fake, X_test_fake, y_train, y_test = train_test_split(train_fake, labels, 
                                                              test_size=.33)

X_train = data[X_train_fake.flatten()]
X_test = data[X_test_fake.flatten()]

np.savez_compressed('train_test/X_train.npz', X_train)
np.savez_compressed('train_test/y_train.npz', y_train)
np.savez_compressed('train_test/X_test.npz', X_test)
np.savez_compressed('train_test/y_test.npz', y_test)
# -*- coding: utf-8 -*-

#==============================================================================
# split X_train to 10-folds cross validation dataset
#==============================================================================

import numpy as np
from sklearn.model_selection import StratifiedKFold

X_train = np.load('train_test_stan/X_train.npz')['arr_0']
y_train = np.load('train_test_stan/y_train.npz')['arr_0']
train_fake = np.arange(X_train.shape[0]*X_train.shape[1]).reshape((X_train.shape[0], X_train.shape[1]))

skf = StratifiedKFold(n_splits=10)

n_k_fold = 0

for train_index, val_index in skf.split(train_fake, y_train):
    np.savez_compressed('10folds_xval/X_train'+str(n_k_fold)+'.npz', np.asarray(X_train[train_index], dtype=np.float32))
    np.savez_compressed('10folds_xval/y_train'+str(n_k_fold)+'.npz', np.asarray(y_train[train_index], dtype=np.float32))
    np.savez_compressed('10folds_xval/X_val'+str(n_k_fold)+'.npz', np.asarray(X_train[val_index], dtype=np.float32))
    np.savez_compressed('10folds_xval/y_val'+str(n_k_fold)+'.npz', np.asarray(y_train[val_index], dtype=np.float32))
    n_k_fold += 1
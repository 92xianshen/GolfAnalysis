# -*- coding: utf-8 -*-

#==============================================================================
# data standardization
#==============================================================================

from __future__ import print_function

import numpy as np
from sklearn.preprocessing import StandardScaler

X_train = np.load('train_test/X_train.npz')['arr_0']
X_test = np.load('train_test/X_test.npz')['arr_0']

X_train_reshape = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_test_reshape = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

stan_scaler = StandardScaler()
stan_scaler.fit(X_train_reshape)

X_train_reshape = stan_scaler.transform(X_train_reshape)
X_test_reshape = stan_scaler.transform(X_test_reshape)

X_train_stan = X_train_reshape.reshape(X_train.shape)
X_test_stan = X_test_reshape.reshape(X_test.shape)

np.savez_compressed('train_test_stan/X_train.npz', X_train_stan)
np.savez_compressed('train_test_stan/X_test.npz', X_test_stan)

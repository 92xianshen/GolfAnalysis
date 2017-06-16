# -*- coding: utf-8 -*-

#==============================================================================
# svm for golf classification
#==============================================================================

from __future__ import print_function

import numpy as np
from sklearn import svm

fname_acc = 'svm_acc.txt'

X_train = np.load('../train_test_stan/X_train.npz')['arr_0']
y_train = np.load('../train_test_stan/y_train.npz')['arr_0']
X_test = np.load('../train_test_stan/X_test.npz')['arr_0']
y_test = np.load('../train_test_stan/y_test.npz')['arr_0']

sensors = {
        'all_sensors': [0, 1, 2, 3, 4, 5, 6, 7], 
        'sg': [0, 1], 
        'acc': [2, 3, 4], 
        'gyro': [5, 6, 7]
        }

for length in [995, 950, 900, 850, 800]:
    for sensor in sensors.keys():
        X_train_select = X_train.copy()[:, sensors[sensor]][:, :, 350:length]
        X_test_select = X_test.copy()[:, sensors[sensor]][:, :, 350:length]
        X_train_select = X_train_select.reshape((X_train_select.shape[0], -1))
        X_test_select = X_test_select.reshape((X_test_select.shape[0], -1))
        
        clf = svm.NuSVC()
        clf.fit(X_train_select, y_train)
        print(clf.score(X_test_select, y_test))
        np.savez_compressed('y_label_'+sensor+'_350_'+str(length)+'_SVM.npz', clf.predict(X_test_select))
        with open(fname_acc, 'a') as f:
            f.writelines(str(length))
            f.writelines(' ')
            f.writelines(str(sensor))
            f.writelines('\n')
            f.writelines(str(clf.score(X_test_select, y_test)))
            f.writelines('\n')

#X_train = X_train.reshape((X_train.shape[0], -1))
#X_test = X_test.reshape((X_test.shape[0], -1))
#
#clf = svm.SVC()
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)
#print(np.mean(np.equal(y_pred, y_test)))
#print(clf.score(X_test, y_test))
#
#clf2 = svm.NuSVC()
#clf2.fit(X_train, y_train)
#y_pred = clf2.predict(X_test)
#print(np.mean(np.equal(y_pred, y_test)))
#print(clf2.score(X_test, y_test))
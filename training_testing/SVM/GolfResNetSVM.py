# -*- coding: utf-8 -*-

#==============================================================================
# svm classifier with cnn feature extraction
#==============================================================================

from __future__ import print_function

import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import Conv1DLayer, InputLayer, MaxPool1DLayer, DenseLayer, dropout
from GolfResNet import GolfResNet

from sklearn import svm

opt = {
       'output_num_units': 19,
       'X_train': '../train_test_stan/X_train.npz',
       'y_train': '../train_test_stan/y_train.npz',
       'X_test': '../train_test_stan/X_test.npz',
       'y_test': '../train_test_stan/y_test.npz',
       'fname_acc': 'svm_resnetfeats_acc.txt',
       'model': 'GolfResNet'
       }

sensors = {
        'all_sensors': [0, 1, 2, 3, 4, 5, 6, 7], 
        'sg': [0, 1], 
        'acc': [2, 3, 4], 
        'gyro': [5, 6, 7]
        }

X_train = np.load(opt['X_train'])['arr_0']
y_train = np.load(opt['y_train'])['arr_0']
X_test = np.load(opt['X_test'])['arr_0']
y_test = np.load(opt['y_test'])['arr_0']

for length in [995, 950, 900, 850, 800]:
    for sensor in sensors.keys():
        X_train_select = X_train.copy()[:, sensors[sensor]][:, :, 350:length]
        X_test_select = X_test.copy()[:, sensors[sensor]][:, :, 350:length]
        
        golf_model = GolfResNet(l_in_shape=(None, len(sensors[sensor]), length-350),
                                l_out_shape=opt['output_num_units'])
        
        init_params_file = '../checkpoints_'+opt['model']+'/golf_model_'+sensor+'_350_'+str(length)+'_'+opt['model']+'_epoch_99.npz'
        with np.load(init_params_file) as f:
            param_values = [f['arr_%d'%i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(golf_model.network['l_out'], param_values)
        

        input_var = T.tensor3('input_var')
        feats = lasagne.layers.get_output(golf_model.network['l_maxpool3'], input_var,
                                  deterministic=True)

        get_feats_fn = theano.function(inputs=[input_var], outputs=feats)
        
        features_train = get_feats_fn(X_train_select.astype(np.float32))
        features_test = get_feats_fn(X_test_select.astype(np.float32))
        features_train = features_train.reshape((X_train_select.shape[0], -1))
        features_test = features_test.reshape((X_test_select.shape[0], -1))
        
        clf = svm.NuSVC()
        clf.fit(features_train, y_train)
        print(clf.score(features_test, y_test))
        with open(opt['fname_acc'], 'a') as f:
            f.writelines(str(length)+' '+str(sensor)+'\n'+str(clf.score(features_test, y_test))+'\n')

#features_train = get_feats_fn(X_train.astype(np.float32))
#features_test = get_feats_fn(X_test.astype(np.float32))
#features_train = features_train.reshape((X_train.shape[0], -1))
#features_test = features_test.reshape((X_test.shape[0], -1))
#
#print(features_train.shape)
#print(features_test.shape)
#
#clf = svm.NuSVC()
#clf.fit(features_train, y_train)
#y_pred = clf.predict(features_test)
#print(np.mean(np.equal(y_pred, y_test)))
#print(clf.score(features_test, y_test))
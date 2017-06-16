# -*- coding: utf-8 -*-

#==============================================================================
# data classification test
# input test.npz
# output labels
# Test result:
# Xavier initialization, 1e-5 weight decay
# acc: 1.0
#==============================================================================

from __future__ import print_function

import time

import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import Conv1DLayer, InputLayer, BatchNormLayer, MaxPool1DLayer, DenseLayer, dropout
from GolfResNet import GolfResNet

# global parameters
opt = {
       'input_shape': (None, 3, 550),
       'output_num_units': 19,
       'X_test': 'train_test_stan/X_test.npz',
       'y_test': 'train_test_stan/y_test.npz',
       'init_params': 'checkpoints_GolfResNet/golf_model_acc_350_900_GolfResNet_epoch_99.npz',
       'sensor_selector': np.asarray([2, 3, 4]),
       'sample_selector': np.arange(350, 900),
       'y_pred_fname': 'y_pred_acc_350_900_GolfResNet.npz',
       'y_label_fname': 'y_label_acc_350_900_GolfResNet.npz'
       }

print('[', time.asctime() ,']',
        'opt:', opt)

# define and build model
print('[', time.asctime() ,']',
        'Defining network...')

golfResNet = GolfResNet(l_in_shape=opt['input_shape'],
                                l_out_shape=opt['output_num_units'])

# compile theano code
print('[', time.asctime() ,']',
        'Defining and compiling model...')

print('[', time.asctime() ,']',
        'Initialing network...')
with np.load(opt['init_params']) as f:
    param_values = [f['arr_%d'%i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(golfResNet.network['l_out'], param_values)

print('[', time.asctime() ,']',
        'Defining theano function...')

input_var = T.tensor3('input_var')
target_var = T.ivector('target_var')

prediction = lasagne.layers.get_output(golfResNet.network['l_out'], input_var,
                                       deterministic=True)
label = T.argmax(prediction, axis=1)
loss = lasagne.objectives.categorical_accuracy(prediction, target_var)
acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

test_fn = theano.function(inputs=[input_var, target_var],
                          outputs=[prediction, label, loss, acc])

print('[', time.asctime() ,']',
        'Starting testing...')

print('[', time.asctime() ,']',
        'Loading test set...')
X_test = np.load(opt['X_test'])['arr_0'][:, opt['sensor_selector']][:, :, opt['sample_selector']]
y_test = np.load(opt['y_test'])['arr_0']

print('[', time.asctime() ,']',
        'Starting testing model...')
test_err = .0
test_acc = .0
test_num = 0
y_predict = []
y_label = []
for i in range(X_test.shape[0]):
    pred, label, err, acc = test_fn(np.asarray([X_test[i]], dtype=np.float32),
                             np.asarray([y_test[i]], dtype=np.uint8))
    y_predict.append(pred)
    y_label.append(label)
    test_err += err
    test_acc += acc
    test_num += 1

print('[', time.asctime() ,']',
        'Test result:')
print('[', time.asctime() ,']',
        'test loss: {}'.format(test_err/test_num))
print('[', time.asctime() ,']',
        'test accuracy: {}'.format(test_acc/test_num))

np.savez_compressed(opt['y_pred_fname'], y_predict)
np.savez_compressed(opt['y_label_fname'], y_label)

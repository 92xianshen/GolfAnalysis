# -*- coding: utf-8 -*-

#==============================================================================
# 20170420
# non-normalization data classification training using all sensor signal sequences
# input: X_train.npz, y_train.npz
# output: loss
# 20170418 Xavier initialization
# 20170418 l2 weight decay strategy
# 20170420 Sensor selection: SG1(0), SG2(1), AccX(3), AccY(4), AccZ(5), GyroX(7), GyroY(8), GyroZ(9)
# 20170420 sample selection: 350-950
#==============================================================================

from __future__ import print_function

import os
import time

import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import Conv1DLayer, InputLayer, BatchNormLayer, MaxPool1DLayer, DenseLayer, dropout

from GolfInception import GolfInception


# global parameters
opt = {
       'niter': 100,
       'input_shape': (None, 8, 450),
       'output_num_units': 19,
       'save_epoch': 100,
       'wt_decay': 1e-5,
       'loss_file': 'train_loss_all_sensors_350_800_GolfInception.txt',
       'X_train': 'train_test_stan/X_train.npz',
       'y_train': 'train_test_stan/y_train.npz',
       'sensor_selector': np.asarray([0, 1, 2, 3, 4, 5, 6, 7]),
       'sample_selector': np.arange(350, 800),
       'checkpoints_folder': 'checkpoints/',
       'model_name': 'golf_model_all_sensors_350_800_GolfInception'
       }

print('[', time.asctime() ,']',
        'opt:', opt)

# define and build model
print('[', time.asctime() ,']',
        'Defining network...')

golfInception = GolfInception(l_in_shape=opt['input_shape'],
                                l_out_shape=opt['output_num_units'])

# compile theano code
print('[', time.asctime() ,']',
        'Defining and compiling model...')

input_var = T.tensor3('input_var')
target_var = T.ivector('target_var')

prediction = lasagne.layers.get_output(golfInception.network['l_out'], input_var)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

# weight decay
weightsl2 = lasagne.regularization.regularize_layer_params(golfInception.network['l_out'],
                                                           lasagne.regularization.l2)
loss += weightsl2*opt['wt_decay']

params = lasagne.layers.get_all_params(golfInception.network['l_out'],
                                       trainable=True)
grad = T.grad(loss, params)
updates = lasagne.updates.adam(grad, params)

val_prediction = lasagne.layers.get_output(golfInception.network['l_out'],
                                           input_var,
                                           deterministic=True)
val_loss = lasagne.objectives.categorical_crossentropy(val_prediction, target_var)
val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1), target_var), dtype=theano.config.floatX)

train_fn = theano.function(inputs=[input_var, target_var],
                           outputs=loss,
                           updates=updates)

val_fn = theano.function(inputs=[input_var, target_var],
                         outputs=[val_loss, val_acc])

print('[', time.asctime() ,']',
        'Loading training data...')
X_train = np.load(opt['X_train'])['arr_0'][:, opt['sensor_selector']][:, :, opt['sample_selector']]
y_train = np.load(opt['y_train'])['arr_0']

print('[', time.asctime() ,']',
        'Removing existing loss record file...')
if os.path.isfile(opt['loss_file']):
    os.remove(opt['loss_file'])
    print('[', time.asctime() ,']',
            'Existing '+opt['loss_file']+' has been removed.')
else:
    print('[', time.asctime() ,']',
            opt['loss_file']+' does not exist.')


print('[', time.asctime() ,']',
        'Starting training model...')
for epoch in range(opt['niter']):
    train_loss = .0
    train_num = 0
    for j in range(X_train.shape[0]):
        train_loss += train_fn(np.asarray([X_train[j]], dtype=np.float32),
                               np.asarray([y_train[j]], dtype=np.uint8))
        train_num += 1
    if (epoch+1)%opt['save_epoch'] == 0:
        np.savez_compressed(opt['checkpoints_folder']+opt['model_name']+'_epoch_'+str(epoch),
                            *lasagne.layers.get_all_param_values(golfInception.network['l_out']))
    with open(opt['loss_file'], 'a') as f:
        f.writelines(str(train_loss/train_num)+'\n')
    print('[', time.asctime(), ']',
            'Epoch {}: train loss {}'.format(epoch, train_loss/train_num))

# -*- coding: utf-8 -*-

#==============================================================================
# 20170601
# hyperparameter selection with 10-folds cross validation using all sensors from 350 to 950 with model GolfVanillaCNN
# input: data/X_train*, data/y_train*, data/X_val*, data/y_val* 
# output: validation accuracy, validation loss, average validation accuracy, average validation loss
# 20170418 Xavier weight initialization????
# 20170418 l2 weight decay strategy
# 20170420 Sensor selection: SG1(0), SG2(1), AccX(2), AccY(3), AccZ(4), GyroX(5), GyroY(6), GyroZ(7)
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

from GolfVanillaCNN import GolfVanillaCNN

# global parameters
opt = {
       'log_file': 'train_10folds_xval_log_all_sensors_350_950_GolfVanillaCNN.txt', 
       'niter': 100,
       'input_shape': (None, 8, 600), 
       'output_num_units': 19, 
       'save_epoch': 100, 
       'val_epoch': 10,
       'wt_decay': 1e-5,
       'loss_file_prefix': 'train_all_sensors_350_950_GolfVanillaCNN_loss_fold_',
       'loss_file_suffix': '.txt',
       'val_acc_file_prefix': 'val_acc_all_sensors_350_950_GolfVanillaCNN_fold_', 
       'val_acc_file_suffix': '.txt',
       'data_dir': 'data/',
       'X_train_prefix': 'X_train', 
       'y_train_prefix': 'y_train', 
       'train_suffix': '.npz', 
       'X_val_prefix': 'X_val', 
       'y_val_prefix': 'y_val', 
       'val_suffix': '.npz', 
       'n_files': 10,
       'sensor_selector': np.asarray([0, 1, 2, 3, 4, 5, 6, 7]),
       'sample_selector': np.arange(350, 950), 
       'checkpoints_folder': 'checkpoints/',
       'model_name': 'golf_model_all_sensor_350_950_GolfVanillaCNN'
       }

# log file
print('[', time.asctime(), ']', 
        'Removing existing loss record file...')
if os.path.isfile(opt['log_file']):
    os.remove(opt['log_file'])
    print('[', time.asctime(), ']', 
            'Existing '+opt['log_file']+' has been removed.')
else:
    print('[', time.asctime()+']', 
            opt['log_file']+' does not exist.')

# output opt
with open(opt['log_file'], 'a') as f:
    f.writelines('['+time.asctime()+'] opt:'+str(opt)+'\n')
print('[', time.asctime() ,']', 
        'opt:', opt)

# 10-folds cross validation
total_start_time = time.time()

with open(opt['log_file'], 'a') as f:
    f.writelines('['+time.asctime()+'] Starting 10-folds cross validation...\n')
print('[', time.asctime() ,']', 
        'Starting 10-folds cross validation...')

total_val_err = []
total_val_acc = []

for i in range(opt['n_files']):
    # time record
    start_time = time.time()
    
    with open(opt['log_file'], 'a') as f:
        f.writelines('['+time.asctime()+'] Starting fold {}...\n'.format(i))
    print('[', time.asctime() ,']',
            'Starting fold {}...'.format(i))
    
    # define and build model
    with open(opt['log_file'], 'a') as f:
        f.writelines('['+time.asctime()+'] Defining network...\n')    
    print('[', time.asctime() ,']', 
            'Defining network...')
#    network = InputLayer(shape=opt['input_shape'])
#    network = Conv1DLayer(network, num_filters=28, filter_size=3,
#                          nonlinearity=lasagne.nonlinearities.rectify, 
#                          W=lasagne.init.GlorotUniform(gain='relu'))
#    network = MaxPool1DLayer(network, pool_size=2)
#    network = Conv1DLayer(network, num_filters=56, filter_size=3,
#                          nonlinearity=lasagne.nonlinearities.rectify, 
#                          W=lasagne.init.GlorotUniform(gain='relu'))
#    network = MaxPool1DLayer(network, pool_size=2)
#    network = Conv1DLayer(network, num_filters=112, filter_size=3, 
#                          nonlinearity=lasagne.nonlinearities.rectify, 
#                          W=lasagne.init.GlorotUniform(gain='relu'))
#    network = MaxPool1DLayer(network, pool_size=2)
#    network = DenseLayer(dropout(network, p=0.5), num_units=512,
#                         nonlinearity=lasagne.nonlinearities.rectify, 
#                         W=lasagne.init.GlorotUniform())
#    network = DenseLayer(dropout(network, p=0.5), num_units=256,
#                         nonlinearity=lasagne.nonlinearities.rectify, 
#                         W=lasagne.init.GlorotUniform())
#    network = DenseLayer(dropout(network, p=0.5), num_units=opt['output_num_units'], 
#                         nonlinearity=lasagne.nonlinearities.softmax, 
#                         W=lasagne.init.GlorotUniform())
    golfVanillaCNN = GolfVanillaCNN(l_in_shape=opt['input_shape'], 
                                  l_out_shape=opt['output_num_units'])
    
    # compile theano code
    with open(opt['log_file'], 'a') as f:
        f.writelines('['+time.asctime()+'] Defining and compiling model...\n')
    print('[', time.asctime() ,']', 
            'Defining and compiling model...')
    
    input_var = T.tensor3('input_var')
    target_var = T.ivector('target_var')
    
    prediction = lasagne.layers.get_output(golfVanillaCNN.network['l_out'], input_var)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    
    # weight decay
    weightsl2 = lasagne.regularization.regularize_layer_params(golfVanillaCNN.network['l_out'], 
                                                               lasagne.regularization.l2)
    loss += weightsl2*opt['wt_decay']
    
    params = lasagne.layers.get_all_params(golfVanillaCNN.network['l_out'], trainable=True)
    grad = T.grad(loss, params)
    updates = lasagne.updates.adam(grad, params)
    
    val_prediction = lasagne.layers.get_output(golfVanillaCNN.network['l_out'], input_var, 
                                               deterministic=True)
    val_loss = lasagne.objectives.categorical_crossentropy(val_prediction, target_var)
    val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1), target_var), dtype=theano.config.floatX)
    
    train_fn = theano.function(inputs=[input_var, target_var], 
                               outputs=loss, 
                               updates=updates)
    
    val_fn = theano.function(inputs=[input_var, target_var],
                             outputs=[val_loss, val_acc])
    
    # remove existing loss record file
    with open(opt['log_file'], 'a') as f:
        f.writelines('['+time.asctime()+'] Removing existing loss record file...\n')
    print('[', time.asctime() ,']', 
        'Removing existing loss record file...')
    if os.path.isfile(opt['loss_file_prefix']+str(i)+opt['loss_file_suffix']):
        os.remove(opt['loss_file_prefix']+str(i)+opt['loss_file_suffix'])
        
    # remove existing validation accuracy file
    with open(opt['log_file'], 'a') as f:
        f.writelines('['+time.asctime()+'] Removing existing validation accuracy record file...\n')
    print('[', time.asctime() ,']', 
        'Removing existing validation accuracy record file...')
    if os.path.isfile(opt['val_acc_file_prefix']+str(i)+opt['val_acc_file_suffix']):
        os.remove(opt['val_acc_file_prefix']+str(i)+opt['val_acc_file_suffix'])
    
    # load data
    with open(opt['log_file'], 'a') as f:
        f.writelines('['+time.asctime()+'] Loading training set fold {}...\n'.format(i))
    print('[', time.asctime(), ']', 
            'Loading training set fold {}...'.format(i))
    X_train = np.load(opt['data_dir']+opt['X_train_prefix']+str(i)+opt['train_suffix'])['arr_0']
    # select sensors and trim data
    X_train = X_train[:, opt['sensor_selector']][:, :, opt['sample_selector']]
    y_train = np.load(opt['data_dir']+opt['y_train_prefix']+str(i)+opt['train_suffix'])['arr_0']
    X_val = np.load(opt['data_dir']+opt['X_val_prefix']+str(i)+opt['val_suffix'])['arr_0']
    # select sensors and trim data
    X_val = X_val[:, opt['sensor_selector']][:, :, opt['sample_selector']]
    y_val = np.load(opt['data_dir']+opt['y_val_prefix']+str(i)+opt['val_suffix'])['arr_0']
    with open(opt['log_file'], 'a') as f:
        f.writelines('['+time.asctime()+'] Dataset shape: X_train.shape={}, y_train.shape={}, X_val.shape={}, y_val.shape={}\n'.format(
                X_train.shape, y_train.shape, X_val.shape, y_val.shape))
    print('[', time.asctime(), ']', 
            'Dataset shape: X_train.shape={}, y_train.shape={}, X_val.shape={}, y_val.shape={}'.format(
                X_train.shape, y_train.shape, X_val.shape, y_val.shape))
    
    # start training model
    with open(opt['log_file'], 'a') as f:
        f.writelines('['+time.asctime()+'] Starting training model...\n')
    print('[', time.asctime(), ']', 
            'Starting training model...')
    for epoch in range(opt['niter']):
        train_loss = .0
        train_num = 0
        for j in range(X_train.shape[0]):
            train_loss += train_fn(np.asarray([X_train[j]], dtype=np.float32), 
                                   np.asarray([y_train[j]], dtype=np.uint8))
            train_num += 1
        with open(opt['loss_file_prefix']+str(i)+opt['loss_file_suffix'], 'a') as f:
            f.writelines(str(train_loss/train_num)+'\n')
        with open(opt['log_file'], 'a') as f:
            f.writelines('['+time.asctime()+'] Epoch {}: train loss {}\n'.format(epoch, train_loss/train_num))
        print('[', time.asctime(), ']',
                'Epoch {}: train loss {}'.format(epoch, train_loss/train_num))
        
        if (epoch+1)%opt['save_epoch'] == 0:
            np.savez_compressed(opt['checkpoints_folder']+opt['model_name'], 
                                *lasagne.layers.get_all_param_values(golfVanillaCNN.network['l_out']))
        if (epoch+1)%opt['val_epoch'] == 0:
            # epoch validation
            with open(opt['log_file'], 'a') as f:
                f.writelines('['+time.asctime()+'] Epoch {} validation\n'.format(epoch))
            print('[', time.asctime(), ']', 
                    'Epoch {} validation'.format(epoch))
            val_err = .0
            val_acc = .0
            val_num = 0
            for j in range(X_val.shape[0]):
                err, acc = val_fn(np.asarray([X_val[j]], dtype=np.float32), 
                          np.asarray([y_val[j]], dtype=np.uint8))
                val_err += err
                val_acc += acc
                val_num += 1
            if epoch == opt['niter']-1:
                total_val_err += [val_err/val_num]
                total_val_acc += [val_acc/val_num]
            
            with open(opt['val_acc_file_prefix']+str(i)+opt['val_acc_file_suffix'], 'a') as f:
                f.writelines(str(val_acc/val_num)+'\n')
            with open(opt['log_file'], 'a') as f:
                f.writelines('['+time.asctime()+'] Epoch {} validation loss: {} validation accuracy: {}\n'.format(epoch, 
                               val_err/val_num, val_acc/val_num))
            print('[', time.asctime(), ']', 
                    'Epoch {} validation loss: {} validation accuracy: {}'.format(epoch, 
                               val_err/val_num, val_acc/val_num))
            
#avg_total_var_err = np.mean(np.asarray(total_val_err))
#avg_total_var_acc = np.mean(np.asarray(total_val_err))
#with open(opt['log_file'], 'a') as f:
#    f.writelines('['+time.asctime()+'] Average total validation loss: {}, average total validation accuracy: {}\n'.format(
#            avg_total_var_err, avg_total_var_acc))
#print('['+time.asctime()+']',
#        'Average total validation loss: {}, average total validation accuracy: {}\n'.format(
#            avg_total_var_err, avg_total_var_acc))
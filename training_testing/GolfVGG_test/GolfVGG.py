# -*- coding: utf-8 -*-

#==============================================================================
# define VGG module
#==============================================================================

from __future__ import print_function

import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, Conv1DLayer, MaxPool1DLayer, DenseLayer, dropout

class GolfVGG:
    def __init__(self, l_in_shape, l_out_shape):
        self.network = {}
        self.network['l_in'] = InputLayer(shape=l_in_shape)
        self.network['l_conv1_1'] = Conv1DLayer(self.network['l_in'], 
                    num_filters=28, 
                    filter_size=3,
                    nonlinearity=lasagne.nonlinearities.rectify, 
                    W=lasagne.init.GlorotUniform(gain='relu'))
        self.network['l_conv1_2'] = Conv1DLayer(self.network['l_conv1_1'], 
                    num_filters=28, 
                    filter_size=3, 
                    nonlinearity=lasagne.nonlinearities.rectify, 
                    W=lasagne.init.GlorotUniform(gain='relu'))
        self.network['l_maxpool1'] = MaxPool1DLayer(self.network['l_conv1_2'], 
                    pool_size=2)
        self.network['l_conv2_1'] = Conv1DLayer(self.network['l_maxpool1'], 
                    num_filters=56, 
                    filter_size=3, 
                    nonlinearity=lasagne.nonlinearities.rectify, 
                    W=lasagne.init.GlorotUniform(gain='relu'))
        self.network['l_conv2_2'] = Conv1DLayer(self.network['l_conv2_1'], 
                    num_filters=56, 
                    filter_size=3, 
                    nonlinearity=lasagne.nonlinearities.rectify, 
                    W=lasagne.init.GlorotUniform(gain='relu'))
        self.network['l_maxpool2'] = MaxPool1DLayer(self.network['l_conv2_2'], 
                    pool_size=2)
        self.network['l_conv3_1'] = Conv1DLayer(self.network['l_maxpool2'], 
                    num_filters=112, 
                    filter_size=3, 
                    nonlinearity=lasagne.nonlinearities.rectify, 
                    W=lasagne.init.GlorotUniform(gain='relu'))
        self.network['l_conv3_2'] = Conv1DLayer(self.network['l_conv3_1'], 
                    num_filters=112, 
                    filter_size=3, 
                    nonlinearity=lasagne.nonlinearities.rectify, 
                    W=lasagne.init.GlorotUniform(gain='relu'))
        self.network['l_conv3_3'] = Conv1DLayer(self.network['l_conv3_2'], 
                    num_filters=112, 
                    filter_size=3, 
                    nonlinearity=lasagne.nonlinearities.rectify, 
                    W=lasagne.init.GlorotUniform(gain='relu'))
        self.network['l_conv3_4'] = Conv1DLayer(self.network['l_conv3_3'], 
                    num_filters=112, 
                    filter_size=3, 
                    nonlinearity=lasagne.nonlinearities.rectify, 
                    W=lasagne.init.GlorotUniform(gain='relu'))
        self.network['l_maxpool3'] = MaxPool1DLayer(self.network['l_conv3_4'], 
                    pool_size=2)
        self.network['l_fc1'] = DenseLayer(dropout(self.network['l_maxpool3'], p=.5), 
                    num_units=256, 
                    nonlinearity=lasagne.nonlinearities.rectify, 
                    W=lasagne.init.GlorotUniform(gain='relu'))
        self.network['l_out'] = DenseLayer(dropout(self.network['l_fc1'], p=.5), 
                    num_units=l_out_shape, 
                    nonlinearity=lasagne.nonlinearities.softmax)
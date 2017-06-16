# -*- coding: utf-8 -*-

import lasagne
from lasagne.layers import InputLayer, Conv1DLayer, MaxPool1DLayer, DenseLayer, dropout

class GolfVanillaCNN:
    def __init__(self, l_in_shape, l_out_shape):
        self.network = {}
        self.network['l_in'] = InputLayer(shape=l_in_shape)
        self.network['l_conv1'] = Conv1DLayer(self.network['l_in'], num_filters=28, filter_size=3,
                    nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
        self.network['l_maxpool1'] = MaxPool1DLayer(self.network['l_conv1'], pool_size=2)
        self.network['l_conv2'] = Conv1DLayer(self.network['l_maxpool1'], num_filters=56, filter_size=3,
                    nonlinearity=lasagne.nonlinearities.rectify)
        self.network['l_maxpool2'] = MaxPool1DLayer(self.network['l_conv2'], pool_size=2)
        self.network['l_conv3'] = Conv1DLayer(self.network['l_maxpool2'], num_filters=112, filter_size=3, 
                    nonlinearity=lasagne.nonlinearities.rectify)
        self.network['l_maxpool3'] = MaxPool1DLayer(self.network['l_conv3'], pool_size=2)
        self.network['l_fc1'] = DenseLayer(dropout(self.network['l_maxpool3'], p=0.5), num_units=512,
                    nonlinearity=lasagne.nonlinearities.rectify)
        self.network['l_fc2'] = DenseLayer(dropout(self.network['l_fc1'], p=0.5), num_units=256,
                    nonlinearity=lasagne.nonlinearities.rectify)
        self.network['l_out'] = DenseLayer(dropout(self.network['l_fc2'], p=0.5), num_units=l_out_shape, 
                    nonlinearity=lasagne.nonlinearities.softmax)
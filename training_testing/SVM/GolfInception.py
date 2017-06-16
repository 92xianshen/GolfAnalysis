# -*- coding: utf-8 -*-

#==============================================================================
# define inception structure
#==============================================================================

from __future__ import print_function

import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, Conv1DLayer, ConcatLayer, MaxPool1DLayer, DenseLayer, dropout

class GolfInception:
    def __init__(self, l_in_shape, l_out_shape):
        def def_inception(layer, no_1=16, no_3r=24, no_3=32, 
                          no_5r=4, no_5=8, no_pool=32):
            l_conv_inc1 = Conv1DLayer(layer, 
                                     num_filters=no_1, 
                                     filter_size=1, 
                                     pad=0, 
                                     stride=1, 
                                     nonlinearity=lasagne.nonlinearities.rectify, 
                                     W=lasagne.init.GlorotUniform(gain='relu'))
            l_conv_inc3r = Conv1DLayer(layer, 
                                      num_filters=no_3r, 
                                      filter_size=1, 
                                      pad=0, 
                                      stride=1, 
                                      nonlinearity=lasagne.nonlinearities.rectify, 
                                      W=lasagne.init.GlorotUniform(gain='relu'))
            l_conv_inc3 = Conv1DLayer(l_conv_inc3r, 
                                       num_filters=no_3, 
                                       filter_size=3, 
                                       pad=1, 
                                       stride=1, 
                                       nonlinearity=lasagne.nonlinearities.rectify, 
                                       W=lasagne.init.GlorotUniform(gain='relu'))
            l_conv_inc5r = Conv1DLayer(layer, 
                                       num_filters=no_5r, 
                                       filter_size=1, 
                                       pad=0, 
                                       stride=1, 
                                       nonlinearity=lasagne.nonlinearities.rectify, 
                                       W=lasagne.init.GlorotUniform(gain='relu'))
            l_conv_inc5 = Conv1DLayer(l_conv_inc5r, 
                                      num_filters=no_5, 
                                      filter_size=5, 
                                      pad=2, 
                                      stride=1, 
                                      nonlinearity=lasagne.nonlinearities.rectify, 
                                      W=lasagne.init.GlorotUniform(gain='relu'))
            l_pool2 = MaxPool1DLayer(layer, 
                                     pool_size=3, 
                                     stride=1, 
                                     pad=1)
            l_conv_inc_pool = Conv1DLayer(l_pool2,
                                          num_filters=no_pool, 
                                          filter_size=1, 
                                          pad=0, 
                                          stride=1, 
                                          nonlinearity=lasagne.nonlinearities.rectify, 
                                          W=lasagne.init.GlorotUniform(gain='relu'))
            l_inc_out = ConcatLayer([l_conv_inc1, l_conv_inc3, l_conv_inc5, l_conv_inc_pool])
            return l_inc_out
        
        self.network = {}
        self.network['l_in'] = InputLayer(shape=l_in_shape)
        self.network['l_conv1'] = Conv1DLayer(self.network['l_in'], 
                                   num_filters=28, 
                                   filter_size=3, 
                                   nonlinearity=lasagne.nonlinearities.rectify, 
                                   W=lasagne.init.GlorotUniform(gain='relu'))
        self.network['l_maxpool1'] = MaxPool1DLayer(self.network['l_conv1'], 
                    pool_size=2)
        self.network['l_inc1'] = def_inception(self.network['l_maxpool1'])
        self.network['l_maxpool2'] = MaxPool1DLayer(self.network['l_inc1'], 
                    pool_size=2)
        self.network['l_inc2'] = def_inception(self.network['l_maxpool2'])
        self.network['l_maxpool3'] = MaxPool1DLayer(self.network['l_inc2'], 
                    pool_size=2)
        self.network['l_inc3'] = def_inception(self.network['l_maxpool3'])
        self.network['l_maxpool4'] = MaxPool1DLayer(self.network['l_inc3'], 
                    pool_size=2)
        self.network['l_fc1'] = DenseLayer(dropout(self.network['l_maxpool4'], p=.5), 
                                  num_units=256, 
                                  nonlinearity=lasagne.nonlinearities.rectify, 
                                  W=lasagne.init.GlorotUniform(gain='relu'))
        self.network['l_out'] = DenseLayer(dropout(self.network['l_fc1'], p=.5), 
                                  num_units=l_out_shape, 
                                  nonlinearity=lasagne.nonlinearities.softmax, 
                                  W=lasagne.init.GlorotUniform(gain='relu'))
        
        
if __name__ == '__main__':
    network = GolfInception(l_in_shape=(None, 13, 1000), 
                        l_out_shape=9)
    print(network.network)
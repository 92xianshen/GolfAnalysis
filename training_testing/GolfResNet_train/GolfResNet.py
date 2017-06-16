# -*- coding: utf-8 -*-

#==============================================================================
# define resnet for golf
#==============================================================================

from __future__ import print_function

import theano

import lasagne
from lasagne.layers import Conv1DLayer, ElemwiseSumLayer, InputLayer, DenseLayer, MaxPool1DLayer, dropout

class GolfResNet:
    def __init__(self, l_in_shape, l_out_shape):
        def projection(l_inp):
            n_filters = l_inp.output_shape[1]*2
            l = Conv1DLayer(l_inp, 
                            num_filters=n_filters, 
                            filter_size=1, 
                            stride=2, 
                            nonlinearity=None, 
                            pad='same', 
                            b=None)
            return l
        
        def filters_increase_dim(l, increase_dim):
            in_num_filters = l.output_shape[1]
            if increase_dim:
                first_stride = 2
                out_num_filters = in_num_filters*2
            else:
                first_stride = 1
                out_num_filters = in_num_filters
            return out_num_filters, first_stride
        
        def res_block_v1(l_inp, increase_dim=False):
            n_filters, first_stride = filters_increase_dim(l_inp, 
                                                           increase_dim=increase_dim)    
            l = Conv1DLayer(l_inp, 
                            num_filters=n_filters, 
                            filter_size=3, 
                            stride=first_stride, 
                            nonlinearity=lasagne.nonlinearities.rectify, 
                            pad='same', 
                            W=lasagne.init.GlorotUniform(gain='relu'))
            l = Conv1DLayer(l, 
                            num_filters=n_filters, 
                            filter_size=3, 
                            stride=1, 
                            nonlinearity=lasagne.nonlinearities.rectify, 
                            pad='same', 
                            W=lasagne.init.GlorotUniform(gain='relu'))
            if increase_dim:
                p = projection(l_inp)
            else:
                p = l_inp
            l = ElemwiseSumLayer([l, p])
            return l
        
        def blockstack(l, n):
            for _ in range(n):
                l = res_block_v1(l)
            return l
        
        self.network = {}
        self.network['l_in'] = InputLayer(shape=l_in_shape)
        self.network['l_conv1'] = Conv1DLayer(self.network['l_in'], 
                    num_filters=28, 
                    filter_size=3, 
                    nonlinearity=lasagne.nonlinearities.rectify, 
                    W=lasagne.init.GlorotUniform(gain='relu'))
        self.network['l_maxpool1'] = MaxPool1DLayer(self.network['l_conv1'], 
                    pool_size=2)
        self.network['l_bs_1'] = blockstack(self.network['l_maxpool1'], 
                    n=1)
        self.network['l_id_1'] = res_block_v1(self.network['l_bs_1'], 
                    increase_dim=True)
        self.network['l_maxpool2'] = MaxPool1DLayer(self.network['l_id_1'], 
                    pool_size=2)
        self.network['l_bs_2'] = blockstack(self.network['l_maxpool2'], 
                    n=1)
        self.network['l_id_2'] = res_block_v1(self.network['l_bs_2'], 
                    increase_dim=True)
        self.network['l_maxpool3'] = MaxPool1DLayer(self.network['l_id_2'], 
                    pool_size=2)
        self.network['l_bs_3'] = blockstack(self.network['l_maxpool3'], 
                    n=1)
        self.network['l_id_3'] = res_block_v1(self.network['l_bs_3'], 
                    increase_dim=True)
        self.network['l_maxpool4'] = MaxPool1DLayer(self.network['l_id_3'], 
                    pool_size=2)
        self.network['l_fc1'] = DenseLayer(dropout(self.network['l_maxpool4'], p=.5), 
                    num_units=256, 
                    nonlinearity=lasagne.nonlinearities.rectify, 
                    W=lasagne.init.GlorotUniform(gain='relu'))
        self.network['l_out'] = DenseLayer(dropout(self.network['l_fc1'], p=.5), 
                    num_units=l_out_shape, 
                    nonlinearity=lasagne.nonlinearities.softmax)
        
if __name__ == '__main__':
    golfResNet = GolfResNet(l_in_shape=(None, 13, 1000), 
                            l_out_shape=9)
    print(golfResNet.network)
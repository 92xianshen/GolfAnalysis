# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

sensors = ['all_sensors', 'sg', 'acc', 'gyro']
seqs = ['350_995', '350_950', '350_900', '350_850', '350_800']

for sensor in sensors:
    for seq in seqs:

#sensor = 'all_sensors'
#seq = '350_995'
        colors = ['navy', 'darkorange', 'cornflowerblue', 'pink', 'teal']
        
        y_test = np.load('../train_test_stan/y_test.npz')['arr_0']
        y_decision_SVM = np.load('../SVM/y_decision_{}_{}_SVM.npz'.format(sensor, seq))['arr_0']
        y_pred_GolfVanillaCNN = np.load('../GolfVanillaCNN_test/y_pred_{}_{}_GolfVanillaCNN.npz'.format(sensor, seq))['arr_0'][:, 0, :]
        y_pred_GolfVGG = np.load('../GolfVGG_test/y_pred_{}_{}_GolfVGG.npz'.format(sensor, seq))['arr_0'][:, 0, :]
        y_pred_GolfInception = np.load('../GolfInception_test/y_pred_{}_{}_GolfInception.npz'.format(sensor, seq))['arr_0'][:, 0, :]
        y_pred_GolfResNet = np.load('../GolfResNet_test/y_pred_{}_{}_GolfResNet.npz'.format(sensor, seq))['arr_0'][:, 0, :]
        
        y_test_bin = label_binarize(y_test, classes=np.arange(19))
        n_classes = 19
        
        precision = dict()
        recall = dict()
        average_precision = dict()
        
        try:
            precision['SVM'], recall['SVM'], _ = precision_recall_curve(y_test_bin.ravel(), 
                     y_decision_SVM.ravel())
            average_precision['SVM'] = average_precision_score(y_test_bin, y_decision_SVM, 
                             average='micro')
        except:
            print 'error!'
        precision['GolfVanillaCNN'], recall['GolfVanillaCNN'], _ = precision_recall_curve(y_test_bin.ravel(), 
                 y_pred_GolfVanillaCNN.ravel())
        average_precision['GolfVanillaCNN'] = average_precision_score(y_test_bin, 
                         y_pred_GolfVanillaCNN, average='micro')
        precision['GolfVGG'], recall['GolfVGG'], _ = precision_recall_curve(y_test_bin.ravel(), 
                 y_pred_GolfVGG.ravel())
        average_precision['GolfVGG'] = average_precision_score(y_test_bin, y_pred_GolfVGG, 
                         average='micro')
        precision['GolfInception'], recall['GolfInception'], _ = precision_recall_curve(y_test_bin.ravel(), 
                 y_pred_GolfInception.ravel())
        average_precision['GolfInception'] = average_precision_score(y_test_bin, y_pred_GolfInception, 
                         average='micro')
        precision['GolfResNet'], recall['GolfResNet'], _ = precision_recall_curve(y_test_bin.ravel(), 
                 y_pred_GolfResNet.ravel())
        average_precision['GolfResNet'] = average_precision_score(y_test_bin, y_pred_GolfResNet, 
                         average='micro')
        
        plt.clf()
        for key, c in zip(precision.keys(), colors):
            plt.plot(recall[key], precision[key], color=c, lw=1, 
                         label='PR curve for {} (area = {:0.2f})'.format(key, average_precision[key]))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR curve of models with {} from sequence {}'.format(sensor, seq))
        plt.legend(loc='lower left')
        plt.savefig('pr_for_{}_{}.png'.format(sensor, seq), dpi=300)
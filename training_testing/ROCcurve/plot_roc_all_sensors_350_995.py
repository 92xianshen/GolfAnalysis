# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

sensors = ['all_sensors']
seqs = ['350_995']
models = ['SVM', 'GolfVanillaCNN', 'GolfVGG', 'GolfInception', 'GolfResNet']

for sensor in sensors:
    for seq in seqs:

#sensor = 'all_sensors'
#seq = '350_995'
        colors = ['navy', 'darkorange', 'cornflowerblue', 'pink', 'teal']
        
        y_test = np.load('../train_test_stan/y_test.npz')['arr_0']
        y_pred = dict()
        for model in models:
            if model == 'SVM':
                y_pred[model] = np.load('../{}/y_decision_{}_{}_{}.npz'.format(model, sensor, seq, model))['arr_0']
            else:
                y_pred[model] = np.load('../{}_test/y_pred_{}_{}_{}.npz'.format(model, sensor, seq, model))['arr_0'][:, 0, :]
        
        y_test_bin = label_binarize(y_test, classes=np.arange(19))
        n_classes = 19
        
        plt.clf()
        for model, c in zip(models, colors):
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            fpr['micro'], tpr['micro'], _ = roc_curve(y_test_bin.ravel(), y_pred[model].ravel())
            roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= n_classes
                fpr['macro'] = all_fpr
                tpr['macro'] = mean_tpr
                roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
            
            plt.plot(fpr['micro'], tpr['micro'], 
                     label='micro-average ROC curve of {} (area = {:.2f})'.format(model, roc_auc['micro']), 
                     color=c, linestyle='--')
            plt.plot(fpr['macro'], tpr['macro'], 
                     label='macro-average ROC curve of {} (area = {:.2f})'.format(model, roc_auc['macro']), 
                     color=c, linestyle=':')
        plt.plot([0, 1], [1, 0], 'k--', color='blue')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve of models with {} from sequence {}'.format(sensor, seq))
        plt.legend(loc='lower right')
        plt.savefig('roc_for_{}_{}.png'.format(sensor, seq), dpi=300)
        
        
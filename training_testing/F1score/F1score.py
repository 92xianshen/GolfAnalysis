# -*- coding: utf-8 -*-

#==============================================================================
# F1 score
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

sensors = ['all_sensors', 'sg', 'acc', 'gyro']
seqs = ['350_995', '350_950', '350_900', '350_850', '350_800']
models = ['SVM', 'GolfVanillaCNN', 'GolfVGG', 'GolfInception', 'GolfResNet']
colors = ['navy', 'darkorange', 'cornflowerblue', 'pink', 'teal']
markers = ['x', 'o', 'v', '^', '*']

y_test = np.load('../train_test_stan/y_test.npz')['arr_0'].astype(np.float32)
y_label = dict()
f1scores = dict()

for sensor in sensors:
    for model in models:
        for seq in seqs:
            if model == 'SVM':
                y_label['{}_{}_{}'.format(sensor, seq, model)] = np.load('../{}/y_label_{}_{}_{}.npz'.format(model, sensor, seq, model))['arr_0'].flatten()
            else:
                y_label['{}_{}_{}'.format(sensor, seq, model)] = np.load('../{}_test/y_label_{}_{}_{}.npz'.format(model, sensor, seq, model))['arr_0'].flatten()
        f1scores['{}_{}_{}'.format(sensor, model, 'micro')] = []
        f1scores['{}_{}_{}'.format(sensor, model, 'macro')] = []
        for seq in seqs:
            f1scores['{}_{}_{}'.format(sensor, model, 'micro')].append(f1_score(y_test, y_label['{}_{}_{}'.format(sensor, seq, model)], average='micro'))
            f1scores['{}_{}_{}'.format(sensor, model, 'macro')].append(f1_score(y_test, y_label['{}_{}_{}'.format(sensor, seq, model)], average='macro'))
    plt.clf()
    for model, c, marker in zip(models, colors, markers):
        plt.plot(f1scores['{}_{}_{}'.format(sensor, model, 'micro')], color=c, marker=marker, 
                          label='micro F1 score of {}'.format(model), linewidth=.5, markersize=4, 
                          linestyle='--')
        plt.plot(f1scores['{}_{}_{}'.format(sensor, model, 'macro')], color=c, marker=marker, 
                          label='macro F1 score of {}'.format(model), linewidth=.5, markersize=4, 
                          linestyle=':')
    plt.grid(True, axis='y', alpha=.5, linewidth=.5)
    plt.xticks(range(5), seqs)
    plt.xlabel('Sequences')
    plt.ylabel('F1 score')
    plt.legend(loc='lower left')
    plt.title('F1 score with {}'.format(sensor))
    plt.savefig('F1score_{}.png'.format(sensor), dpi=300)
        
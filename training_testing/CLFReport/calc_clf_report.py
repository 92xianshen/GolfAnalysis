# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

sensors = ['all_sensors', 'sg', 'acc', 'gyro']
seqs = ['350_995', '350_950', '350_900', '350_850', '350_800']
models = ['SVM', 'GolfVanillaCNN', 'GolfVGG', 'GolfInception', 'GolfResNet']

y_test = np.load('../train_test_stan/y_test.npz')['arr_0'].astype(np.float32)
y_label = dict()
clf_reports = dict()
conf_mats = dict()

for sensor in sensors:
    for model in models:
        for seq in seqs:
            if model == 'SVM':
                y_label['{}_{}_{}'.format(sensor, seq, model)] = np.load('../{}/y_label_{}_{}_{}.npz'.format(model, sensor, seq, model))['arr_0'].flatten()
            else:
                y_label['{}_{}_{}'.format(sensor, seq, model)] = np.load('../{}_test/y_label_{}_{}_{}.npz'.format(model, sensor, seq, model))['arr_0'].flatten()

for sensor in sensors:
    for model in models:
        for seq in seqs:
            clf_reports['{}_{}_{}'.format(sensor, seq, model)] = classification_report(y_test, y_label['{}_{}_{}'.format(sensor, seq, model)])
            conf_mats['{}_{}_{}'.format(sensor, seq, model)] = confusion_matrix(y_test, y_label['{}_{}_{}'.format(sensor, seq, model)])
            
with open('clf_reports.txt', 'w') as f:
    for sensor in sensors:
        for model in models:
            for seq in seqs:
                f.writelines('{}_{}_{}\n{}\n\n\n'.format(sensor, model, seq, clf_reports['{}_{}_{}'.format(sensor, seq, model)]))
                
with open('conf_mats.txt', 'w') as f:
    for sensor in sensors:
        for model in models:
            for seq in seqs:
                f.writelines('{}_{}_{}\n{}\n\n\n'.format(sensor, model, seq, conf_mats['{}_{}_{}'.format(sensor, seq, model)]))
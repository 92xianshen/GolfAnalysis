# -*- coding: utf-8 -*-

import numpy as np

model = 'GolfInception'

sensors = ['all_sensors', 'sg', 'acc', 'gyro']
seqs = ['995', '950', '900', '850', '800']
y_test = np.load('../train_test_stan/y_test.npz')['arr_0']
fname_acc = 'acc_'+model+'.txt'

for sensor in sensors:
    for seq in seqs:
        fname = 'y_label_'+sensor+'_350_'+seq+'_'+model+'.npz'
        y_label = np.load(fname)['arr_0']
        acc = np.mean(np.equal(y_test.flatten(), y_label.flatten().astype(np.float32)))
        with open(fname_acc, 'a') as f:
            f.writelines(sensor+'\t350_'+str(seq)+'\t'+str(acc)+'\n')

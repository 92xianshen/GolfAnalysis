# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

model = 'GolfVGG'

sensors = ['all_sensors', 'sg', 'acc', 'gyro']
seqs = ['350_800', '350_850', '350_900', '350_950', '350_995']
y_test = np.load('../train_test_stan/y_test.npz')['arr_0']
colors = ['navy', 'darkorange', 'cornflowerblue', 'pink']
markers = ['x', 'o', 'v', '^']
fname_acc = 'acc_'+model+'.txt'

plt.clf()
for sensor, color, marker in zip(sensors, colors, markers):
    acc = []
    for seq in seqs:
        fname = 'y_label_'+sensor+'_'+seq+'_'+model+'.npz'
        y_label = np.load(fname)['arr_0']
        acc.append(np.mean(np.equal(y_test.flatten(), y_label.flatten().astype(np.float32))))
    plt.plot(acc, color=color, marker=marker, label=sensor,
             linewidth=.5, markersize=4)
plt.grid(True, axis='y', alpha=.5, linewidth=.5)
plt.xticks(range(5), seqs)
plt.title('Test accuracy of model {}'.format(model))
plt.xlabel('Sequences')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('test_acc_{}.png'.format(model), dpi=300)

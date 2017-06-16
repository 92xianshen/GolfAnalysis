# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

sensor = 'acc'
models = ['SVM', 'GolfVanillaCNN', 'GolfVGG', 'GolfInception', 'GolfResNet']
seqs = ['350_800', '350_850', '350_900', '350_950', '350_995']
y_test = np.load('../train_test_stan/y_test.npz')['arr_0'].astype(np.float32)
colors = ['navy', 'darkorange', 'cornflowerblue', 'pink', 'teal']
markers = ['x', 'o', 'v', '^', '*']

plt.clf()
for model, color, marker in zip(models, colors, markers):
    acc = []
    for seq in seqs:
        if model == 'SVM':
            fname = '../{}/y_label_{}_{}_{}.npz'.format(model, sensor, seq, model)
        else:
            fname = '../{}_test/y_label_{}_{}_{}.npz'.format(model, sensor, seq, model)
        y_label = np.load(fname)['arr_0'].flatten()
        acc.append(np.mean(np.equal(y_test.flatten(), y_label.flatten().astype(np.float32))))
    plt.plot(acc, color=color, marker=marker, label=model,
             linewidth=.5, markersize=4)
plt.grid(True, axis='y', alpha=.5, linewidth=.5)
plt.xticks(range(5), seqs)
plt.title('Test accuracy of {}'.format(sensor))
plt.xlabel('Sequences')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('test_acc_{}.png'.format(sensor), dpi=300)

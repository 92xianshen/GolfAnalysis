# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

gyro_comp = np.loadtxt('gyro_comp.txt')
gyro_comp = gyro_comp.T

colors = ['navy', 'darkorange', 'cornflowerblue', 'pink', 'teal']
markers = ['x', 'o', 'v', '^', '*']
labels = ['SVM', 'SVM with Vanilla CNN features', 'SVM with VGG-like CNN features',
          'SVM with Inception-based CNN features', 'SVM with Residual-based CNN features']

for i in range(5):
    plt.plot(gyro_comp[i], color=colors[i], marker=markers[i], label=labels[i],
             linewidth=.5, markersize=4)
plt.grid(True, axis='y', alpha=.5, linewidth=.5)
plt.xticks(range(5), ('350-995', '350-950', '350-900', '350-850', '350-800'))
plt.title('CNN-feature-based SVM classification gyrouracy with the gyroscope sensor')
plt.xlabel('Sequences')
plt.ylabel('gyrouracy')
plt.legend()
#plt.show()
plt.savefig('gyro_svm_comp.png', dpi=300)

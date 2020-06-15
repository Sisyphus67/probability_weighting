# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 00:25:09 2020

@author: MK-PC-LML
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

x = np.linspace(-10, 10, num=5000)

# Gaussians location and scale parameters
s1 = 1
l1 = 0
df = 1
# CDFs
DO = stat.norm.pdf(x, l1, s1)
DM = stat.t.pdf(x, df)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 3.5))
fig.tight_layout(pad=3)
#fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.5))
#fig.tight_layout(pad=3)

#t-distribution PDF plot
axes.set_xlim((-4.5, 4.5))
axes.plot(x, DO, 'r', lineWidth='2', label=r'$p(x)$')
axes.plot(x, DM, 'b', lineWidth='2', label=r'$w(x)$')
axes.set_xlabel(r'$x$', fontsize='x-large')
axes.set_ylabel(r'PDFs', fontsize='x-large')
axes.legend(loc='upper left', fontsize='x-large')
axes.set_title('Different Shapes: Gaussian and $t$-distribution', fontsize='x-large')


plt.savefig("./../diff_shapes_Gauss_t.pdf", bbox_inches='tight')
plt.show()
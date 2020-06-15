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
s2 = 2
l1 = 0
l2 = 0
# CDFs
DO = stat.norm.pdf(x, l1, s1)
DM = stat.norm.pdf(x, l2, s2)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.5))
fig.tight_layout(pad=3)

#PDF plot
axes[0].set_xlim((-4.5, 4.5))
axes[0].plot(x, DO, 'r', lineWidth='2', label=r'$p(x)$')
axes[0].plot(x, DM, 'b', lineWidth='2', label=r'$w(x)$')
axes[0].set_xlabel(r'$x$', fontsize='x-large')
axes[0].set_ylabel(r'PDFs', fontsize='x-large')
axes[0].legend(loc='upper left', fontsize='x-large')
axes[0].set_title('Different Scales', fontsize='x-large')

#plot arrows
#xs=[-2.5,-2,-1.5,-1,-.5,0]
#for i in xs:
#    axes[0].arrow(i,stat.norm.pdf(i, l1, s1), 0, stat.norm.pdf(i, l2, s2) - stat.norm.pdf(i, l1, s1), head_width=0.16, head_length=0.02, length_includes_head=True, fc='k', ec='k', zorder=10)

# Gaussians location and scale parameters
s1 = 1
s2 = 1
l1 = 0
l2 = 1
# CDFs
DO = stat.norm.pdf(x, l1, s1)
DM = stat.norm.pdf(x, l2, s2)

#PDF plot
axes[1].set_xlim((-4.5, 4.5))
axes[1].plot(x, DO, 'r', lineWidth='2', label=r'$p(x)$')
axes[1].plot(x, DM, 'b', lineWidth='2', label=r'$w(x)$')
axes[1].set_xlabel(r'$x$', fontsize='x-large')
axes[1].set_ylabel(r'PDFs', fontsize='x-large')
axes[1].legend(loc='upper left', fontsize='x-large')
axes[1].set_title('Different Locations', fontsize='x-large')

plt.savefig("./../2GaussianPDFs2Scales2Locations.pdf", bbox_inches='tight')
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 00:28:24 2020

@author: MK-PC-LML
"""

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

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 3.5))
fig.tight_layout(pad=3)

#PDF plot
axes.set_xlim((-4.5, 4.5))
axes.plot(x, DO, 'r', lineWidth='2', label=r'$p(x)$')
axes.plot(x, DM, 'b', lineWidth='2', label=r'$w(x)$')
axes.set_title('Different Scales')


plt.savefig("./../2GaussianPDFs2Scales.pdf", bbox_inches='tight')
plt.show()
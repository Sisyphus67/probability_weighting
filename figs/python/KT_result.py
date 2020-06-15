# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 00:05:26 2020

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat

#Tversky and Kahneman's function
def s(x, beta):
    s = np.power(x, beta) / np.power((np.power(x, beta) + np.power(1 - x, beta)), 1. / beta)
    return s

# CDFs
x = np.linspace(-100, 100, num=5000)
DO = stat.norm.cdf(x, loc=0, scale=1)
DM_loc = stat.norm.cdf(x, loc=0.23, scale=1)
DM_scale = stat.norm.cdf(x, loc=0, scale=1.64)
DM_loc_scale = stat.norm.cdf(x, loc=0.23, scale=1.64)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 3.5))
fig.tight_layout(pad=3)

# plot Tversky and Kahneman (1992) approximated decision weights CDF
beta = 0.65
axes.plot(DO, DO, 'r', lineWidth='2', label=r'$F_p$')
axes.axvline(x=0.5, LineStyle='--')
axes.plot(DO, s(DO, beta), 'b', lineWidth='2', label=r'$F_w$')
axes.set_title('Tversky and Kahneman (1992)', fontsize='x-large')
axes.set_xlabel(r'CDF $F_p$', fontsize='x-large')
axes.set_ylabel(r'CDFs', fontsize='x-large')
axes.set_xticks(np.arange(0, 1.1, step=0.2))
axes.set_yticks(np.arange(0, 1.1, step=0.2))
#axes.legend(loc='upper left', fontsize='x-large')

fig.savefig("./../KT_result_nolegend.pdf", bbox_inches='tight')
plt.show()
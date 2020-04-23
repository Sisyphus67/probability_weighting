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

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
fig.tight_layout(pad=3)

#plot location
axes[0, 0].plot(DO, DO,'r', lineWidth='2', label=r'$F_p$')
axes[0, 0].axvline(x=0.5,LineStyle='--')
axes[0, 0].plot(DO, DM_loc, 'b', lineWidth='2',label=r'$F_w$')
axes[0, 0].set_title('Location')
axes[0, 0].set_xlabel(r'CDF $F_p$')
axes[0, 0].set_ylabel(r'CDFs')
axes[0, 0].set_xticks(np.arange(0, 1.1, step=0.2))
axes[0, 0].set_yticks(np.arange(0, 1.1, step=0.2))
axes[0, 0].legend(loc='upper left', fontsize='x-small')

#plot scale
axes[0, 1].plot(DO, DO,'r', lineWidth='2', label=r'$F_p$')
axes[0, 1].axvline(x=0.5,LineStyle='--')
axes[0, 1].plot(DO, DM_scale, 'b', lineWidth='2',label=r'$F_w$')
axes[0, 1].set_title('Scale')
axes[0, 1].set_xlabel(r'CDF $F_p$')
axes[0, 1].set_ylabel(r'CDFs')
axes[0, 1].set_xticks(np.arange(0, 1.1, step=0.2))
axes[0, 1].set_yticks(np.arange(0, 1.1, step=0.2))
axes[0, 1].legend(loc='upper left', fontsize='x-small')

#plot scale
axes[1, 0].plot(DO, DO,'r', lineWidth='2', label=r'$F_p$')
axes[1, 0].axvline(x=0.5,LineStyle='--')
axes[1, 0].plot(DO, DM_loc_scale, 'b', lineWidth='2',label=r'$F_w$')
axes[1, 0].set_title('Location and scale')
axes[1, 0].set_xlabel(r'CDF $F_p$')
axes[1, 0].set_ylabel(r'CDFs')
axes[1, 0].set_xticks(np.arange(0, 1.1, step=0.2))
axes[1, 0].set_yticks(np.arange(0, 1.1, step=0.2))
axes[1, 0].legend(loc='upper left', fontsize='x-small')

# plot Tversky and Kahneman (1992) approximated decision weights CDF
beta = 0.65
axes[1, 1].plot(DO, DO, 'r', lineWidth='2', label=r'$F_p$')
axes[1, 1].axvline(x=0.5, LineStyle='--')
axes[1, 1].plot(DO, s(DO, beta), 'b', lineWidth='2', label=r'$F_w$')
axes[1, 1].set_title('Tversky and Kahneman (1992)')
axes[1, 1].set_xlabel(r'CDF $F_p$')
axes[1, 1].set_ylabel(r'CDFs')
axes[1, 1].set_xticks(np.arange(0, 1.1, step=0.2))
axes[1, 1].set_yticks(np.arange(0, 1.1, step=0.2))
axes[1, 1].legend(loc='upper left', fontsize='x-small')

fig.savefig("./../Gauss_scale_location_both_KT.pdf", bbox_inches='tight')
plt.show()
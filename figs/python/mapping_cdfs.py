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
DO = stat.norm.cdf(x, l1, s1)
DM = stat.norm.cdf(x, l2, s2)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.5))
fig.tight_layout(pad=3)

#CDF plot
axes[0].set_xlim((-4.5, 4.5))
axes[0].plot(x, DO, 'r', lineWidth='2', label=r'$F_{p}(x)$')
axes[0].axvline(x=0, LineStyle='--')
axes[0].plot(x, DM, 'b', lineWidth='2', label=r'$F_{w}(x)$')

#plot arrows
xs=[-2,-1.5,-1,-.5,.5,1,1.5,2]
#xs=[-2]
#xs=[-2,-1.5]
#xs=[-2,-1.5,-1]
#xs=[-2,-1.5,-1,-.5]
#xs=[-2,-1.5,-1,-.5,.5]
#xs=[-2,-1.5,-1,-.5,.5,1]
#xs=[-2,-1.5,-1,-.5,.5,1,1.5]

#for i in xs:
#    axes[0].arrow(i,stat.norm.cdf(i, l1, s1), 0, stat.norm.cdf(i, l2, s2) - stat.norm.cdf(i, l1, s1), head_width=0.15, head_length=0.045, length_includes_head=True, fc='k', ec='k', zorder=10)

axes[0].set_xlabel(r'$x$', fontsize='x-large')
axes[0].set_ylabel(r'CDFs', fontsize='x-large')
axes[0].set_xticks(np.arange(-4, 5, step=2))
axes[0].set_yticks(np.arange(0, 1.1, step=0.2))
axes[0].legend(loc='upper left', fontsize='x-large')

#w(p) plot
axes[1].plot(DO, DO, 'r', lineWidth='2', label=r'$F_p(F_p)$')
#axes[1].annotate('$F_p(F_p)$', fontsize=14, xy=(0.2,0.2), xytext=(0.6, 0.2), arrowprops=dict(facecolor='black', shrink=0.1), color='red')
axes[1].axvline(x=0.5, LineStyle='--')
axes[1].plot(DO, DM, 'b', lineWidth='2', label=r'$F_w(F_p)$')

# plot arrows
#for i in xs:
#    axes[1].arrow(stat.norm.cdf(i, l1, s1),stat.norm.cdf(i, l1, s1), 0, stat.norm.cdf(i, l2, s2) - stat.norm.cdf(i, l1, s1), head_width=0.018, head_length=0.045, length_includes_head=True, fc='k', ec='k', zorder=10)

axes[1].set_xlabel(r'CDF $F_p$', fontsize='x-large')
axes[1].set_ylabel(r'CDFs', fontsize='x-large')
axes[1].set_xticks(np.arange(0, 1.1, step=0.2))
axes[1].set_yticks(np.arange(0, 1.1, step=0.2))
axes[1].legend(loc='upper left', fontsize='x-large')

plt.savefig("./../mapping_cdfs_noarrows.pdf", bbox_inches='tight')
#plt.savefig("./../mapping_cdfs.pdf", bbox_inches='tight')
#plt.savefig("./../mapping_cdfs_1arrow.pdf", bbox_inches='tight')
#plt.savefig("./../mapping_cdfs_2arrows.pdf", bbox_inches='tight')
#plt.savefig("./../mapping_cdfs_3arrows.pdf", bbox_inches='tight')
#plt.savefig("./../mapping_cdfs_4arrows.pdf", bbox_inches='tight')
#plt.savefig("./../mapping_cdfs_5arrows.pdf", bbox_inches='tight')
#plt.savefig("./../mapping_cdfs_6arrows.pdf", bbox_inches='tight')
#plt.savefig("./../mapping_cdfs_7arrows.pdf", bbox_inches='tight')
plt.show()
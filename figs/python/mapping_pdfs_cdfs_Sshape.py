import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

x = np.linspace(-10, 10, num=5000)

# Gaussians location and scale parameters
s1 = 1
s2 = 0.5
l1 = 0
l2 = 0
# PDFs
DO = stat.norm.pdf(x, l1, s1)
DM = stat.norm.pdf(x, l2, s2)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
fig.tight_layout(pad=3)

#PDF plot
axes[0, 0].set_xlim((-4.5, 4.5))
axes[0, 0].set_ylim((-0.05, 0.85))
axes[0, 0].plot(x, DO, 'r', lineWidth='2', label=r'$p(x)$')
axes[0, 0].plot(x, DM, 'b', lineWidth='2', label=r'$w(x)$')

#plot arrows
xs=[-2,-1.5,-1,-.4,0]
for i in xs:
    axes[0, 0].arrow(i,stat.norm.pdf(i, l1, s1), 0, stat.norm.pdf(i, l2, s2) - stat.norm.pdf(i, l1, s1), head_width=0.16, head_length=0.045, length_includes_head=True, fc='k', ec='k', zorder=10)

axes[0, 0].set_xlabel(r'$x$')
axes[0, 0].set_ylabel(r'PDFs')
axes[0, 0].set_xticks(np.arange(-4, 5, step=2))
axes[0, 0].set_yticks(np.arange(0, 0.9, step=0.1))
axes[0, 0].legend(loc='upper left', fontsize='x-small')

#w(p) plot
axes[0, 1].set_ylim((-0.05, 0.85))
axes[0, 1].plot(DO, DO, 'r', lineWidth='2', label=r'$p$')
axes[0, 1].plot(DO, DM, 'b', lineWidth='2', label=r'$w$')

for i in xs:
    axes[0, 1].arrow(stat.norm.pdf(i, l1, s1),stat.norm.pdf(i, l1, s1), 0, stat.norm.pdf(i, l2, s2) - stat.norm.pdf(i, l1, s1), head_width=0.008, head_length=0.045, length_includes_head=True, fc='k', ec='k', zorder=10)

axes[0, 1].set_xlabel(r'PDF $p$')
axes[0, 1].set_ylabel(r'PDFs')
axes[0, 1].set_xticks(np.arange(0, .5, step=0.1))
axes[0, 1].set_yticks(np.arange(0, 0.9, step=0.1))
axes[0, 1].legend(loc='upper left', fontsize='x-small')

#CDF plot

# CDFs
DO = stat.norm.cdf(x, l1, s1)
DM = stat.norm.cdf(x, l2, s2)

axes[1, 0].set_xlim((-4.5, 4.5))
axes[1, 0].plot(x, DO, 'r', lineWidth='2', label=r'$F_{p}(x)$')
axes[1, 0].axvline(x=0, LineStyle='--')
axes[1, 0].plot(x, DM, 'b', lineWidth='2', label=r'$F_{w}(x)$')

#plot arrows
xs=[-1.5,-1,-.5,.5,1,1.5]

for i in xs:
    axes[1, 0].arrow(i,stat.norm.cdf(i, l1, s1), 0, stat.norm.cdf(i, l2, s2) - stat.norm.cdf(i, l1, s1), head_width=0.15, head_length=0.045, length_includes_head=True, fc='k', ec='k', zorder=10)

axes[1, 0].set_xlabel(r'$x$')
axes[1, 0].set_ylabel(r'CDFs')
axes[1, 0].set_xticks(np.arange(-4, 5, step=2))
axes[1, 0].set_yticks(np.arange(0, 1.1, step=0.2))
axes[1, 0].legend(loc='upper left', fontsize='x-small')

#w(p) plot
axes[1, 1].plot(DO, DO, 'r', lineWidth='2', label=r'$F_p(F_p)$')
axes[1, 1].annotate('$F_p(F_p)$', fontsize=14, xy=(0.32, 0.32), xytext=(0.6, 0.32), arrowprops=dict(facecolor='black', shrink=0.1), color='red')
axes[1, 1].axvline(x=0.5, LineStyle='--')
axes[1, 1].plot(DO, DM, 'b', lineWidth='2', label=r'$F_w(F_p)$')

# plot arrows
for i in xs:
    axes[1, 1].arrow(stat.norm.cdf(i, l1, s1),stat.norm.cdf(i, l1, s1), 0, stat.norm.cdf(i, l2, s2) - stat.norm.cdf(i, l1, s1), head_width=0.018, head_length=0.045, length_includes_head=True, fc='k', ec='k', zorder=10)

axes[1, 1].set_xlabel(r'CDF $F_p$')
axes[1, 1].set_ylabel(r'CDFs')
axes[1, 1].set_xticks(np.arange(0, 1.1, step=0.2))
axes[1, 1].set_yticks(np.arange(0, 1.1, step=0.2))
axes[1, 1].legend(loc='upper left', fontsize='x-small')

plt.savefig("./../mapping_pdfs_cdfs_Sshape.pdf", bbox_inches='tight')
plt.show()
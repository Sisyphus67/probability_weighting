import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import quad
import scipy.stats as stat

# SCfigure=True
# if SCfigure:
#     font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 15}
#     plt.rc('font', **font)

T = 10 # arbitrary number of observations

xx = np.linspace(0.0, 1.0, num=1000)
x = np.linspace(-100, 100, num=5000)
# Gaussians location and scale parameters
l = 0
s = 1
# PDFs
DO = stat.norm.pdf(x, l, s)
DM = DO+np.sqrt(DO/T)
# CDFs
DM_normalization=scipy.integrate.trapz(DM,x)
DM=DM/DM_normalization

DO_CDF=np.append(0,scipy.integrate.cumtrapz(DO,x))
DM_CDF=np.append(0,scipy.integrate.cumtrapz(DM,x))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
fig.tight_layout(pad=3)

#plot PDFs
axes[0,0].plot(x,DO,'r', label=r'$p(x)$',linewidth=2)
axes[0,0].plot(x,DM,'b', label=r'$w(x)$',linewidth=2)
axes[0,0].set_xlabel(r'$x$')
axes[0,0].set_ylabel(r'PDFs')
axes[0,0].set_xlim((-4,4))
axes[0,0].set_title('Gaussian PDF')
axes[0,0].legend(loc='upper left',fontsize='x-small')
axes[0,0].set_xticks(np.arange(-4, 5, step=2))
axes[0,0].set_yticks(np.arange(0, 0.6, step=0.1))

#plot CDFs
axes[0,1].plot(DO_CDF,DO_CDF,'r', label=r'$F_p$',linewidth=2)
axes[0,1].axvline(x=0.5,LineStyle='--')
axes[0,1].plot(DO_CDF,DM_CDF,'b', label=r'$F_w$',linewidth=2)
axes[0,1].set_xlabel(r'$F_p$')
axes[0,1].set_ylabel(r'CDFs')
axes[0,1].set_title('Gaussian inverse S')
axes[0,1].legend(loc='upper left',fontsize='x-small')
axes[0,1].set_xticks(np.arange(0, 1.1, step=0.2))
axes[0,1].set_yticks(np.arange(0, 1.1, step=0.2))
#axes[1].xlabel(r'$x$')
#axes[1].ylabel(r'PDFs $p$ and $w$')


# Student-t shape parameter
s = 2
# PDFs
DO = stat.t.pdf(x, s)
DM = DO+np.sqrt(DO/T)
# CDFs
DM_normalization=scipy.integrate.trapz(DM,x)
DM=DM/DM_normalization

DO_CDF=np.append(0,scipy.integrate.cumtrapz(DO,x))
DM_CDF=np.append(0,scipy.integrate.cumtrapz(DM,x))

#plot PDFs
axes[1,0].plot(x,DO,'r', label=r'$p(x)$',linewidth=2)
axes[1,0].plot(x,DM,'b', label=r'$w(x)$',linewidth=2)
axes[1,0].set_xlabel(r'$x$')
axes[1,0].set_ylabel(r'PDFs')
axes[1,0].set_xlim((-4,4))
axes[1,0].set_title('t distribution PDF')
axes[1,0].legend(loc='upper left',fontsize='x-small')
axes[1,0].set_xticks(np.arange(-4, 5, step=2))
axes[1,0].set_yticks(np.arange(0, 0.6, step=0.1))

#plot CDFs
axes[1,1].plot(DO_CDF,DO_CDF,'r', label=r'$F_p$',linewidth=2)
axes[1,1].axvline(x=0.5,LineStyle='--')
axes[1,1].plot(DO_CDF,DM_CDF,'b', label=r'$F_w$',linewidth=2)
axes[1,1].set_xlabel(r'$F_p$')
axes[1,1].set_ylabel(r'CDFs')
axes[1,1].set_title('t distribution inverse S')
axes[1,1].legend(loc='upper left',fontsize='x-small')
axes[1,1].set_xticks(np.arange(0, 1.1, step=0.2))
axes[1,1].set_yticks(np.arange(0, 1.1, step=0.2))
#axes[1].xlabel(r'$x$')
#axes[1].ylabel(r'PDFs $p$ and $w$')

plt.savefig("./../square_root_error.pdf", bbox_inches='tight')
plt.show()
plt.clf()
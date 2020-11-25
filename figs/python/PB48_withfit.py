import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import scipy.optimize as opt

def TKfit(x, gamma):
    return (x ** gamma) / (( x ** gamma + (1 - x) ** gamma) ** (1/gamma))

# load data
PB1948 = np.genfromtxt('PB1948_2.csv', delimiter=',')

xx = np.linspace(0.0, 1.0, num=1000)

#plot area
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.5))
fig.tight_layout(pad=3)

popt_c, pcov_c = opt.curve_fit(TKfit, PB1948[:, 0], PB1948[:, 1])
wp4 = TKfit(xx, popt_c[0])
print(popt_c[0])

#plot
axes[0].plot(xx, xx, '--k', label=r'')
axes[0].plot(PB1948[:, 0], PB1948[:, 1], 'ok', markersize=5)
axes[0].set_xlabel(r'Probability')
axes[0].set_ylabel(r'Weight')
axes[0].set_xticks(np.arange(0, 1.1, step=0.2))
axes[0].set_yticks(np.arange(0, 1.1, step=0.2))

axes[1].plot(xx, xx, '--k', label=r'')
axes[1].axvline(x=0.5, LineStyle='--')
axes[1].plot(xx, wp4, color='m', lineWidth='2', label=r"$F^{TK}_w = F_p^{\gamma} / ( F_p^{\gamma} + (1-F_p)^{\gamma})^{(1/\gamma)}$")
axes[1].set_xlabel(r'Probability')
axes[1].set_ylabel(r'Weight')
axes[1].set_xticks(np.arange(0, 1.1, step=0.2))
axes[1].set_yticks(np.arange(0, 1.1, step=0.2))

plt.savefig("./../PB48_withfit.pdf", bbox_inches='tight')
plt.show()
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats as stat

start = time.time()

# Lattimore et al. parameters
deltaTK = 0.66993
gammaTK = 0.59479
deltaTF = 0.77437
gammaTF = 0.68924

# Gaussian parameters
locTK = 0.39888
scaTK = 1.4913
locTF = 0.23012
scaTF = 1.3755

# Student's-t parameters
nuTK = 0.39034
deltatTK = 0.28711
nuTF = 0.46699
deltatTF = 0.17304

# load data
TF1995 = np.genfromtxt('TF1995.csv', delimiter=',')
TK1992 = np.genfromtxt('TK1992.csv', delimiter=',')

xx = np.linspace(0.0, 1.0, num=1000)
x = np.linspace(-1000, 1000, num=10000)
p = stat.norm.cdf(x, 0, 1)
pT = stat.nct.cdf(x, 1, 0)

#plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
fig.tight_layout(pad=3.5)

#TK1992
wp1 = (deltaTK * (xx ** gammaTK)) / (deltaTK * (xx ** gammaTK) + ((1 - xx) ** gammaTK))
wp2 = stat.norm.cdf(x, locTK, scaTK)
wp3 = stat.nct.cdf(x, df = nuTK, nc = deltatTK, loc = 0, scale = 1)

axes[0].plot(xx, xx, '--k', label=r'')
axes[0].axvline(x=0.5, LineStyle='--')
axes[0].plot(xx, wp1, 'b', lineWidth='3', label=r'$w(p)=\delta p^{\gamma} / (\delta p^{\gamma} + (1-p)^{\gamma})$')
axes[0].plot(p, wp2, 'r', lineWidth='3', label=r'Gaussian model')
axes[0].plot(pT, wp3, color='0.5', lineWidth='3', label=r"Student's-t model")
axes[0].plot(TK1992[:, 0], TK1992[:, 1], 'ok', markersize=5, label=r'Experiment')
axes[0].set_title('Tversky & Kahneman (1992)')
axes[0].set_xlabel(r'CDF $p$')
axes[0].set_ylabel(r'decision weights CDF $w(p)$')
axes[0].set_xticks(np.arange(0, 1.1, step=0.2))
axes[0].set_yticks(np.arange(0, 1.1, step=0.2))
axes[0].legend(loc='upper left', fontsize='x-small')

#TF1995
wp1 = (deltaTF * (xx ** gammaTF)) / (deltaTF * (xx ** gammaTF) + ((1 - xx) ** gammaTF))
wp2 = stat.norm.cdf(x, locTF, scaTF)
wp3 = stat.nct.cdf(x, df = nuTF, nc = deltatTF, loc = 0, scale = 1)

axes[1].plot(xx, xx, '--k', label=r'')
axes[1].axvline(x=0.5, LineStyle='--')
axes[1].plot(xx, wp1, 'b', lineWidth='3', label=r'$w(p)=\delta p^{\gamma} / (\delta p^{\gamma} + (1-p)^{\gamma})$')
axes[1].plot(p, wp2, 'r', lineWidth='3', label=r'Gaussian model')
axes[1].plot(pT, wp3, color='0.5', lineWidth='3', label=r"Student's-t model")
axes[1].plot(TF1995[:, 0], TF1995[:, 1], 'ok', markersize=5, label=r'Experiment')
axes[1].set_title('Tversky & Fox (1995)')
axes[1].set_xlabel(r'CDF $p$')
axes[1].set_ylabel(r'decision weights CDF $w(p)$')
axes[1].set_xticks(np.arange(0, 1.1, step=0.2))
axes[1].set_yticks(np.arange(0, 1.1, step=0.2))
axes[1].legend(loc='upper left', fontsize='x-small')

plt.savefig("./../TK_TF_fit.pdf", bbox_inches='tight')
plt.show()
plt.clf()

end = time.time()
print(end - start)
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats as stat

start = time.time()

# number of standard errors for envelopes
nse = 1

# Lattimore et al. parameters
deltaTK = 0.66993; SEdTK = 0.07
gammaTK = 0.59479; SEgTK = 0.06
deltaTF = 0.77437; SEdTF = 0.02
gammaTF = 0.68924; SEgTF = 0.02

# Gaussian parameters
locTK = 0.39888; SElTK = 0.13
scaTK = 1.4913; SEsTK = 0.2
locTF = 0.23012; SElTF = 0.03
scaTF = 1.3755; SEsTF = 0.06

# Student's-t parameters
nuTK = 0.39034; SEnTK = 0.06
deltatTK = 0.28711; SEdtTK = 0.08
nuTF = 0.46699; SEnTF = 0.05
deltatTF = 0.17304; SEdtTF = 0.03

# load data
TF1995 = np.genfromtxt('TF1995.csv', delimiter=',')
TK1992 = np.genfromtxt('TK1992.csv', delimiter=',')

xx = np.linspace(0.0, 1.0, num=1000)
x = np.linspace(-1000, 1000, num=10000)
p = stat.norm.cdf(x, 0, 1)
pT = stat.nct.cdf(x, 1, 0)

#plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.5))
fig.tight_layout(pad=3)

#TK1992
wp1 = (deltaTK * (xx ** gammaTK)) / (deltaTK * (xx ** gammaTK) + ((1 - xx) ** gammaTK))
dtmp1 = deltaTK + nse * SEdTK; gtmp1 = gammaTK + nse * SEgTK
dtmp2 = deltaTK - nse * SEdTK; gtmp2 = gammaTK + nse * SEgTK
dtmp3 = deltaTK + nse * SEdTK; gtmp3 = gammaTK - nse * SEgTK
dtmp4 = deltaTK - nse * SEdTK; gtmp4 = gammaTK - nse * SEgTK
wp1m1 = (dtmp1 * (xx ** gtmp1)) / (dtmp1 * (xx ** gtmp1) + ((1 - xx) ** gtmp1))
wp1m2 = (dtmp2 * (xx ** gtmp2)) / (dtmp2 * (xx ** gtmp2) + ((1 - xx) ** gtmp2))
wp1m3 = (dtmp3 * (xx ** gtmp3)) / (dtmp3 * (xx ** gtmp3) + ((1 - xx) ** gtmp3))
wp1m4 = (dtmp4 * (xx ** gtmp4)) / (dtmp4 * (xx ** gtmp4) + ((1 - xx) ** gtmp4))
wp2 = stat.norm.cdf(x, locTK, scaTK)
wp1g1 = stat.norm.cdf(x, locTK + nse * SElTK, scaTK + nse * SEsTK)
wp1g2 = stat.norm.cdf(x, locTK - nse * SElTK, scaTK + nse * SEsTK)
wp1g3 = stat.norm.cdf(x, locTK + nse * SElTK, scaTK - nse * SEsTK)
wp1g4 = stat.norm.cdf(x, locTK - nse * SElTK, scaTK - nse * SEsTK)
wp3 = stat.nct.cdf(x, df = nuTK, nc = deltatTK, loc = 0, scale = 1)
wp1t1 = stat.nct.cdf(x, df = nuTK + nse * SEnTK, nc = deltatTK + nse * SEdtTK, loc = 0, scale = 1)
wp1t2 = stat.nct.cdf(x, df = nuTK - nse * SEnTK, nc = deltatTK + nse * SEdtTK, loc = 0, scale = 1)
wp1t3 = stat.nct.cdf(x, df = nuTK + nse * SEnTK, nc = deltatTK - nse * SEdtTK, loc = 0, scale = 1)
wp1t4 = stat.nct.cdf(x, df = nuTK - nse * SEnTK, nc = deltatTK - nse * SEdtTK, loc = 0, scale = 1)

axes[0].fill_between(xx, np.minimum(wp1m4, np.minimum(wp1m3, np.minimum(wp1m2, wp1m1))), np.maximum(wp1m4, np.maximum(wp1m3, np.maximum(wp1m2, wp1m1))), facecolor='b', alpha=0.25, interpolate=True)
axes[0].fill_between(p, np.minimum(wp1g4, np.minimum(wp1g3, np.minimum(wp1g2, wp1g1))), np.maximum(wp1g4, np.maximum(wp1g3, np.maximum(wp1g2, wp1g1))), facecolor='r', alpha=0.25, interpolate=True)
axes[0].fill_between(pT, np.minimum(wp1t4, np.minimum(wp1t3, np.minimum(wp1t2, wp1t1))), np.maximum(wp1t4, np.maximum(wp1t3, np.maximum(wp1t2, wp1t1))), facecolor='0.5', alpha=0.25, interpolate=True)
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
dtmp1 = deltaTF + nse * SEdTF; gtmp1 = gammaTF + nse * SEgTF
dtmp2 = deltaTF - nse * SEdTF; gtmp2 = gammaTF + nse * SEgTF
dtmp3 = deltaTF + nse * SEdTF; gtmp3 = gammaTF - nse * SEgTF
dtmp4 = deltaTF - nse * SEdTF; gtmp4 = gammaTF - nse * SEgTF
wp1m1 = (dtmp1 * (xx ** gtmp1)) / (dtmp1 * (xx ** gtmp1) + ((1 - xx) ** gtmp1))
wp1m2 = (dtmp2 * (xx ** gtmp2)) / (dtmp2 * (xx ** gtmp2) + ((1 - xx) ** gtmp2))
wp1m3 = (dtmp3 * (xx ** gtmp3)) / (dtmp3 * (xx ** gtmp3) + ((1 - xx) ** gtmp3))
wp1m4 = (dtmp4 * (xx ** gtmp4)) / (dtmp4 * (xx ** gtmp4) + ((1 - xx) ** gtmp4))
wp2 = stat.norm.cdf(x, locTF, scaTF)
wp1g1 = stat.norm.cdf(x, locTF + nse * SElTF, scaTF + nse * SEsTF)
wp1g2 = stat.norm.cdf(x, locTF - nse * SElTF, scaTF + nse * SEsTF)
wp1g3 = stat.norm.cdf(x, locTF + nse * SElTF, scaTF - nse * SEsTF)
wp1g4 = stat.norm.cdf(x, locTF - nse * SElTF, scaTF - nse * SEsTF)
wp3 = stat.nct.cdf(x, df = nuTF, nc = deltatTF, loc = 0, scale = 1)
wp1t1 = stat.nct.cdf(x, df = nuTF + nse * SEnTF, nc = deltatTF + nse * SEdtTF, loc = 0, scale = 1)
wp1t2 = stat.nct.cdf(x, df = nuTF - nse * SEnTF, nc = deltatTF + nse * SEdtTF, loc = 0, scale = 1)
wp1t3 = stat.nct.cdf(x, df = nuTF + nse * SEnTF, nc = deltatTF - nse * SEdtTF, loc = 0, scale = 1)
wp1t4 = stat.nct.cdf(x, df = nuTF - nse * SEnTF, nc = deltatTF - nse * SEdtTF, loc = 0, scale = 1)

axes[1].fill_between(xx, np.minimum(wp1m4, np.minimum(wp1m3, np.minimum(wp1m2, wp1m1))), np.maximum(wp1m4, np.maximum(wp1m3, np.maximum(wp1m2, wp1m1))), facecolor='b', alpha=0.25, interpolate=True)
axes[1].fill_between(p, np.minimum(wp1g4, np.minimum(wp1g3, np.minimum(wp1g2, wp1g1))), np.maximum(wp1g4, np.maximum(wp1g3, np.maximum(wp1g2, wp1g1))), facecolor='r', alpha=0.25, interpolate=True)
axes[1].fill_between(pT, np.minimum(wp1t4, np.minimum(wp1t3, np.minimum(wp1t2, wp1t1))), np.maximum(wp1t4, np.maximum(wp1t3, np.maximum(wp1t2, wp1t1))), facecolor='0.5', alpha=0.25, interpolate=True)
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

plt.savefig("./TK_TF_fit.pdf", bbox_inches='tight')
plt.show()
plt.clf()

end = time.time()
print(end - start)
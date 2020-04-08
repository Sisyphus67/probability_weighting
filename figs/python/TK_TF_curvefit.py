import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats as stat
import scipy.optimize as opt

start = time.time()

def larrimore(x, delta, gamma):
    return delta * (x ** gamma) / (delta * (x ** gamma) + (1 - x) ** gamma)

def gaussianfit(x, l, s):
    return stat.norm.cdf(stat.norm.ppf(x, loc=0, scale=1), loc=l, scale=s)

def nctfit(x, l, s):
    return stat.nct.cdf(stat.nct.ppf(x, df=1, nc=0), df=s, nc=l)

# load data
TF1995 = np.genfromtxt('TF1995.csv', delimiter=',')
TK1992 = np.genfromtxt('TK1992.csv', delimiter=',')

# number of standard errors for envelopes
nse = 2

xx = np.linspace(0.0, 1.0, num=1000)

#plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.5))
fig.tight_layout(pad=3)

xx = np.linspace(0.0, 1.0, num=1000)

#TK1992

#fit TK1992
popt_l, pcov_l = opt.curve_fit(larrimore, TK1992[:, 0], TK1992[:, 1])
popt_g, pcov_g = opt.curve_fit(gaussianfit, TK1992[:, 0], TK1992[:, 1])
popt_t, pcov_t = opt.curve_fit(nctfit, TK1992[:, 0], TK1992[:, 1])
perr_l = np.sqrt(np.diag(pcov_l))
perr_g = np.sqrt(np.diag(pcov_g))
perr_t = np.sqrt(np.diag(pcov_t))

wp1 = larrimore(xx, popt_l[0], popt_l[1])
wp1m1 = larrimore(xx, popt_l[0] + nse * perr_l[0], popt_l[1] + nse * perr_l[1])
wp1m2 = larrimore(xx, popt_l[0] - nse * perr_l[0], popt_l[1] + nse * perr_l[1])
wp1m3 = larrimore(xx, popt_l[0] + nse * perr_l[0], popt_l[1] - nse * perr_l[1])
wp1m4 = larrimore(xx, popt_l[0] - nse * perr_l[0], popt_l[1] - nse * perr_l[1])
wp2 = gaussianfit(xx, popt_g[0], popt_g[1])
wp2m1 = gaussianfit(xx, popt_g[0] + nse * perr_g[0], popt_g[1] + nse * perr_g[1])
wp2m2 = gaussianfit(xx, popt_g[0] - nse * perr_g[0], popt_g[1] + nse * perr_g[1])
wp2m3 = gaussianfit(xx, popt_g[0] + nse * perr_g[0], popt_g[1] - nse * perr_g[1])
wp2m4 = gaussianfit(xx, popt_g[0] - nse * perr_g[0], popt_g[1] - nse * perr_g[1])
wp3 = nctfit(xx, popt_t[0], popt_t[1])
wp3m1 = nctfit(xx, popt_t[0] + nse * perr_t[0], popt_t[1] + nse * perr_t[1])
wp3m2 = nctfit(xx, popt_t[0] - nse * perr_t[0], popt_t[1] + nse * perr_t[1])
wp3m3 = nctfit(xx, popt_t[0] + nse * perr_t[0], popt_t[1] - nse * perr_t[1])
wp3m4 = nctfit(xx, popt_t[0] - nse * perr_t[0], popt_t[1] - nse * perr_t[1])

axes[0].fill_between(xx, np.minimum(wp1m4, np.minimum(wp1m3, np.minimum(wp1m2, wp1m1))), np.maximum(wp1m4, np.maximum(wp1m3, np.maximum(wp1m2, wp1m1))), facecolor='b', alpha=0.25, interpolate=True)
axes[0].fill_between(xx, np.minimum(wp2m4, np.minimum(wp2m3, np.minimum(wp2m2, wp2m1))), np.maximum(wp2m4, np.maximum(wp2m3, np.maximum(wp2m2, wp2m1))), facecolor='r', alpha=0.25, interpolate=True)
axes[0].fill_between(xx, np.minimum(wp3m4, np.minimum(wp3m3, np.minimum(wp3m2, wp3m1))), np.maximum(wp3m4, np.maximum(wp3m3, np.maximum(wp3m2, wp3m1))), facecolor='0.5', alpha=0.25, interpolate=True)
axes[0].plot(xx, xx, '--k', label=r'')
axes[0].axvline(x=0.5, LineStyle='--')
axes[0].plot(xx, wp1, 'b', lineWidth='2', label=r'$F_w=\delta F_p^{\gamma} / (\delta F_p^{\gamma} + (1-F_p)^{\gamma})$')
axes[0].plot(xx, wp2, 'r', lineWidth='2', label=r'Gaussian model')
axes[0].plot(xx, wp3, color='0.5', lineWidth='2', label=r"Student's-t model")
axes[0].plot(TK1992[:, 0], TK1992[:, 1], 'ok', markersize=5, label=r'Experiment')
axes[0].set_title('Tversky & Kahneman (1992)')
axes[0].set_xlabel(r'CDF $F_p$')
axes[0].set_ylabel(r'decision weights CDF $F_w$')
axes[0].set_xticks(np.arange(0, 1.1, step=0.2))
axes[0].set_yticks(np.arange(0, 1.1, step=0.2))
axes[0].legend(loc='upper left', fontsize='x-small')

#TF1995

#fit TF1995
popt_l, pcov_l = opt.curve_fit(larrimore, TF1995[:, 0], TF1995[:, 1])
popt_g, pcov_g = opt.curve_fit(gaussianfit, TF1995[:, 0], TF1995[:, 1])
popt_t, pcov_t = opt.curve_fit(nctfit, TF1995[:, 0], TF1995[:, 1])
perr_l = np.sqrt(np.diag(pcov_l))
perr_g = np.sqrt(np.diag(pcov_g))
perr_t = np.sqrt(np.diag(pcov_t))

wp1 = larrimore(xx, popt_l[0], popt_l[1])
wp1m1 = larrimore(xx, popt_l[0] + nse * perr_l[0], popt_l[1] + nse * perr_l[1])
wp1m2 = larrimore(xx, popt_l[0] - nse * perr_l[0], popt_l[1] + nse * perr_l[1])
wp1m3 = larrimore(xx, popt_l[0] + nse * perr_l[0], popt_l[1] - nse * perr_l[1])
wp1m4 = larrimore(xx, popt_l[0] - nse * perr_l[0], popt_l[1] - nse * perr_l[1])
wp2 = gaussianfit(xx, popt_g[0], popt_g[1])
wp2m1 = gaussianfit(xx, popt_g[0] + nse * perr_g[0], popt_g[1] + nse * perr_g[1])
wp2m2 = gaussianfit(xx, popt_g[0] - nse * perr_g[0], popt_g[1] + nse * perr_g[1])
wp2m3 = gaussianfit(xx, popt_g[0] + nse * perr_g[0], popt_g[1] - nse * perr_g[1])
wp2m4 = gaussianfit(xx, popt_g[0] - nse * perr_g[0], popt_g[1] - nse * perr_g[1])
wp3 = nctfit(xx, popt_t[0], popt_t[1])
wp3m1 = nctfit(xx, popt_t[0] + nse * perr_t[0], popt_t[1] + nse * perr_t[1])
wp3m2 = nctfit(xx, popt_t[0] - nse * perr_t[0], popt_t[1] + nse * perr_t[1])
wp3m3 = nctfit(xx, popt_t[0] + nse * perr_t[0], popt_t[1] - nse * perr_t[1])
wp3m4 = nctfit(xx, popt_t[0] - nse * perr_t[0], popt_t[1] - nse * perr_t[1])

axes[1].fill_between(xx, np.minimum(wp1m4, np.minimum(wp1m3, np.minimum(wp1m2, wp1m1))), np.maximum(wp1m4, np.maximum(wp1m3, np.maximum(wp1m2, wp1m1))), facecolor='b', alpha=0.25, interpolate=True)
axes[1].fill_between(xx, np.minimum(wp2m4, np.minimum(wp2m3, np.minimum(wp2m2, wp2m1))), np.maximum(wp2m4, np.maximum(wp2m3, np.maximum(wp2m2, wp2m1))), facecolor='r', alpha=0.25, interpolate=True)
axes[1].fill_between(xx, np.minimum(wp3m4, np.minimum(wp3m3, np.minimum(wp3m2, wp3m1))), np.maximum(wp3m4, np.maximum(wp3m3, np.maximum(wp3m2, wp3m1))), facecolor='0.5', alpha=0.25, interpolate=True)
axes[1].plot(xx, xx, '--k', label=r'')
axes[1].axvline(x=0.5, LineStyle='--')
axes[1].plot(xx, wp1, 'b', lineWidth='2', label=r'$F_w=\delta F_p^{\gamma} / (\delta F_p^{\gamma} + (1-F_p)^{\gamma})$')
axes[1].plot(xx, wp2, 'r', lineWidth='2', label=r'Gaussian model')
axes[1].plot(xx, wp3, color='0.5', lineWidth='2', label=r"Student's-t model")
axes[1].plot(TF1995[:, 0], TF1995[:, 1], 'ok', markersize=5, label=r'Experiment')
axes[1].set_title('Tversky & Fox (1995)')
axes[1].set_xlabel(r'CDF $F_p$')
axes[1].set_ylabel(r'decision weights CDF $F_w$')
axes[1].set_xticks(np.arange(0, 1.1, step=0.2))
axes[1].set_yticks(np.arange(0, 1.1, step=0.2))
axes[1].legend(loc='upper left', fontsize='x-small')

plt.savefig("./../TK_TF_curvefit.pdf", bbox_inches='tight')
#plt.show()
#plt.clf()

end = time.time()
print(end - start)
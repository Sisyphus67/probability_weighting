import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat

x = np.linspace(-100, 100, num=5000)

# CDFs
DO_CDF = stat.t.cdf(x, loc=0.2, df=3)
DM_CDF = stat.t.cdf(x, loc=0.35, df=1)

plt.figure(figsize=(4.5, 3.5))
plt.tight_layout(pad=3)

plt.plot(DO_CDF, DO_CDF, 'r', label=r'$F_p$', linewidth=2)
plt.axvline(x=0.5, LineStyle='--')
plt.plot(DO_CDF, DM_CDF, 'b', label=r'$F_w$', linewidth=2)
plt.xlabel(r'CDF $F_p$')
plt.ylabel(r'CDFs')
plt.legend(loc='upper left', fontsize='x-small')
plt.xticks(np.arange(0, 1.1, step=0.2))
plt.yticks(np.arange(0, 1.1, step=0.2))

plt.savefig("./../Student-t.pdf", bbox_inches='tight')
plt.show()
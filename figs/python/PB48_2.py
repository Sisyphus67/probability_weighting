import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import scipy.optimize as opt

# load data
PB1948 = np.genfromtxt('PB1948_2.csv', delimiter=',')

xx = np.linspace(0.0, 1.0, num=1000)

#plot area
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 3.5))
fig.tight_layout(pad=3)

#plot
axes.plot(xx, xx, '--k', label=r'')
axes.plot(PB1948[:, 0], PB1948[:, 1], 'ok', markersize=5)
axes.set_xlabel(r'Probability')
axes.set_ylabel(r'Weight')
axes.set_xticks(np.arange(0, 1.1, step=0.2))
axes.set_yticks(np.arange(0, 1.1, step=0.2))

plt.savefig("./../PB48_2.pdf", bbox_inches='tight')
plt.show()
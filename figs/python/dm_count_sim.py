# Check python version
import sys
print("Python is:", sys.version)

# Import matplotlib and numpy
import matplotlib
import matplotlib.pyplot as plt
print("Matplotlib is:", matplotlib.__version__)

import numpy as np
print("numpy is:", np.__version__)

# import random
from scipy.stats import norm
from scipy.stats import t
# from scipy.stats import triang

# number and length of time series
N=1000
T=100

# histogram parameters
x_min=-5
x_max=5
n_bins=100
bin_width=(x_max-x_min)/n_bins

# define figure and axes
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
fig.tight_layout(pad=3)


# loop through normal and t distributions
for i in range(0, 2): # incremented for loop
    if i==0:
        # normal distribution
        l = (x_max+x_min)/2 # location
        s = (x_max-x_min)/10 # scale
        dist = norm(loc=l, scale=s)
    
    else:
        # t distribution
        df = 1.5 # shape
        l = (x_max+x_min)/2 # location
        s = (x_max-x_min)/20 # scale
        dist = t(df, loc=l, scale=s)
        
        # triangular distribution
        # c = 0.5 # shape (centre)
        # l = x_min # location
        # s = x_max-x_min # scale
        # dist = triang(c, loc=l, scale=s)
        
    # generate random variables
    X=dist.rvs(size=[N,T])
    
    # count
    n=np.empty((N,n_bins))
    for series in range(0,N):
        n[series][:]=np.histogram(X[series][:],bins=n_bins,range=(x_min,x_max))[0]
        x_bins=np.histogram(X[series][:],bins=n_bins,range=(x_min,x_max))[1] # bin edges
    x_l=x_bins[:-1] # left edges
    x_r=x_bins[1:] # right edges
    x_bin_c=(x_l+x_r)/2 # bin centres
    
    # find DM's count, mean count, and uncertainty
    dm_count=n[0] # DM's count
    mean_count=np.mean(n,0)# sample mean count across DMs
    uncertainty=np.std(n,0) # uncertainty in count
    count=dm_count # decide which count to use
    
    # construct decision weights and normalize
    phat=count/(T*bin_width) # estimated density
    wraw=phat+uncertainty/(T*bin_width) # unnormalised decision weight
    wmass=np.sum(wraw)*bin_width # unnormalised mass
    w=wraw/wmass # normalised decision weight density
    
    # find decision weight CDF
    Fw=np.cumsum(w)*bin_width
    
    # find reference CDF
    Fp=dist.cdf(x_bins) # all bin edges
    Fp=(Fp-Fp[0])/(Fp[-1]-Fp[0]) # normalise Fp over truncated range
    Fw=np.concatenate(([0],Fw)) # add left bin edge to Fw
    
    # plot
    #fig=plt.figure()
    #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.5))
    #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    #ax=fig.add_axes([0,0,1,1])
    axes[i, 0].hist(x_bins[:-1], x_bins, weights=wraw, color='r', label='')
    axes[i, 0].hist(x_bins[:-1], x_bins, weights=phat, color='b', label='')
    # axes[i, 0].legend(loc=2)
    # axes[1].hist(x_bins[:-1], x_bins, weights=uncertainty/count,color='r',label='relative uncertainty')
    # axes[1].legend(loc=2)
    axes[i, 1].plot(Fp,Fw,'b',label='')
    axes[i, 1].plot(Fp,Fp,'r',label='')
    plt.savefig("./../dm_count_sim.pdf", bbox_inches='tight')
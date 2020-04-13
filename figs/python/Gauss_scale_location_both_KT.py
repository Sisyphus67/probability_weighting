#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:26:27 2020

@author: obp48
"""
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import invgamma
from scipy.stats import expon
from scipy.special import gamma
import numpy as np
import scipy
from scipy.integrate import quad

# SCfigure=True
# if SCfigure:
#     font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 15}
#     plt.rc('font', **font)
    
pos_x=np.linspace(0.01,10,1000)
all_x=np.linspace(-10,10,1000)

def gauss_p(x):
    p=1/(np.sqrt(2*np.pi*(g_scale_p**2)))*np.exp(-np.power((x-g_loc_p),2)/(2*(g_scale_p**2)))
    return p

def cdf_gauss_p(x):
    cdf=quad(gauss_p,-10,x)
    return cdf

def gauss_w(x):
    w=1/(np.sqrt(2*np.pi*(g_scale_w**2)))*np.exp(-np.power((x-g_loc_w),2)/(2*(g_scale_w**2)))
    return w

def cdf_gauss_w(x):
    cdf=quad(gauss_w,-10,x)
    return cdf

#Gaussian
g_loc_p=0
g_loc_w=0
g_scale_p=1
g_scale_w=1.64

#plotting CDFs
#Gauss
i=0
cp=np.zeros((2,all_x.size))
for x in all_x:
    cp[0][i]=x
    cp[1][i]=cdf_gauss_p(x)[0]
    i=i+1

i=0
cw=np.zeros((2,all_x.size))
for x in all_x:
    cw[0][i]=x
    cw[1][i]=cdf_gauss_w(x)[0]
    i=i+1

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
fig.tight_layout(pad=3)

axes[0, 1].plot(cp[1],cp[1],'r', lineWidth='2', label=r'$F_p$')
axes[0, 1].axvline(x=0.5,LineStyle='--')
axes[0, 1].plot(cp[1],cw[1],'b', lineWidth='2',label=r'$F_w$')
axes[0, 1].set_title('Scale')
axes[0, 1].set_xlabel(r'CDF $F_p$')
axes[0, 1].set_ylabel(r'CDFs')
axes[0, 1].set_xticks(np.arange(0, 1.1, step=0.2))
axes[0, 1].set_yticks(np.arange(0, 1.1, step=0.2))
axes[0, 1].legend(loc='upper left', fontsize='x-small')

#Gaussian
g_loc_p=0
g_loc_w=.23
g_scale_p=1
g_scale_w=1

#plotting CDFs
#Gauss
i=0
cp=np.zeros((2,all_x.size))
for x in all_x:
    cp[0][i]=x
    cp[1][i]=cdf_gauss_p(x)[0]
    i=i+1

i=0
cw=np.zeros((2,all_x.size))
for x in all_x:
    cw[0][i]=x
    cw[1][i]=cdf_gauss_w(x)[0]
    i=i+1

axes[0, 0].plot(cp[1],cp[1],'r',lineWidth='2', label=r'$F_p$')
axes[0, 0].axvline(x=0.5,LineStyle='--')
axes[0, 0].plot(cp[1],cw[1],'b',lineWidth='2',label=r'$F_w$')
axes[0, 0].set_title('Location')
axes[0, 0].set_xlabel(r'CDF $F_p$')
axes[0, 0].set_ylabel(r'CDFs')
axes[0, 0].set_xticks(np.arange(0, 1.1, step=0.2))
axes[0, 0].set_yticks(np.arange(0, 1.1, step=0.2))
axes[0, 0].legend(loc='upper left', fontsize='x-small')

#Gaussian
g_loc_p=0
g_loc_w=.23
g_scale_p=1
g_scale_w=1.64

#plotting CDFs
i=0
cp=np.zeros((2,all_x.size))
for x in all_x:
    cp[0][i]=x
    cp[1][i]=cdf_gauss_p(x)[0]
    i=i+1

i=0
cw=np.zeros((2,all_x.size))
for x in all_x:
    cw[0][i]=x
    cw[1][i]=cdf_gauss_w(x)[0]
    i=i+1

axes[1, 0].plot(cp[1], cp[1], 'r', lineWidth='2', label=r'$F_p$')
axes[1, 0].axvline(x=0.5, LineStyle='--')
axes[1, 0].plot(cp[1], cw[1], 'b', lineWidth='2', label=r'$F_w$')
axes[1, 0].set_title('Location and scale')
axes[1, 0].set_xlabel(r'CDF $F_p$')
axes[1, 0].set_ylabel(r'CDFs')
axes[1, 0].set_xticks(np.arange(0, 1.1, step=0.2))
axes[1, 0].set_yticks(np.arange(0, 1.1, step=0.2))
axes[1, 0].legend(loc='upper left', fontsize='x-small')


pos_x=np.linspace(0.0,1,100)
all_x=np.linspace(-10,10,1000)
beta=.65
#Tversky and Kahneman's function
def s(x,beta):
    s=np.power(x,beta)/np.power((np.power(x,beta)+np.power(1-x,beta)),1./beta)
    return s

axes[1, 1].plot(pos_x,pos_x,'r',lineWidth='2', label=r'$F_p$')
axes[1, 1].axvline(x=0.5,LineStyle='--')
axes[1, 1].plot(pos_x,s(pos_x,beta),'b', lineWidth='2',label=r'$F_w$')
axes[1, 1].set_title('Tversky and Kahneman (1992)')
axes[1, 1].set_xlabel(r'CDF $F_p$')
axes[1, 1].set_ylabel(r'CDFs')
axes[1, 1].set_xticks(np.arange(0, 1.1, step=0.2))
axes[1, 1].set_yticks(np.arange(0, 1.1, step=0.2))
axes[1, 1].legend(loc='upper left', fontsize='x-small')

fig.savefig("./../Gauss_scale_location_both_KT.pdf", bbox_inches='tight')
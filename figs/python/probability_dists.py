#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:26:27 2020

@author: obp48
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import quad


SCfigure=True
if SCfigure:
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}
    plt.rc('font', **font)
    


sigma=.2
sigma1=0.13
mu=.1
dt=.1

def re(p):
    ss=sigma*sigma
    s1s=sigma1*sigma1
    re=np.power(p,ss/(ss+s1s))\
    *np.power(np.sqrt(2*np.pi*ss*dt),ss/(ss+s1s))/np.sqrt(2*np.pi*(ss+s1s)*dt)    
    return re

def q(lnr):
    ss=sigma*sigma
    s1s=sigma1*sigma1
    q=1/(np.sqrt(2*np.pi*(s1s+ss)*dt))*np.exp(-np.power((lnr-mu*dt),2)/(2*(s1s+ss)*dt))
    return q

def p(lnr):
    ss=sigma*sigma
    #s1s=sigma1*sigma1
    p=1/(np.sqrt(2*np.pi*(ss)*dt))*np.exp(-np.power((lnr-mu*dt),2)/(2*(ss)*dt))
    return p

def cdfp(x):
    cdfp=quad(p,-10,x)
    return cdfp

def cdfq(x):
    cdfq=quad(q,-10,x)
    return cdfq

#Tversky and Kahneman's function
def s(x,beta):
    s=np.power(x,beta)/np.power((np.power(x,beta)+np.power(1-x,beta)),1./beta)
    return s

lnr=np.arange(-.5,.5,.001)
y=np.arange(0,1,.001)
prob=p(lnr)
reweighted=q(lnr)

i=0
cp=np.zeros((2,lnr.size))
for x in lnr:
    cp[0][i]=x
    cp[1][i]=cdfp(x)[0]
    i=i+1

i=0
cq=np.zeros((2,lnr.size))
for x in lnr:
    cq[0][i]=x
    cq[1][i]=cdfq(x)[0]
    i=i+1

p=np.arange(0,6.3,.01)
prosp=re(p)    

#Plotting distributions
plt.plot(lnr,prob,'b', label=r'PDF $p$ modeled by DO')
plt.plot(lnr,reweighted,'r', label=r'PDF $w$ modeled by DM')
plt.ylabel(r'PDF $p$')
plt.xlabel(r'log return')
plt.legend(loc='upper left',fontsize='x-small')

plt.savefig("./../probability_dists.pdf", bbox_inches='tight')
plt.show()
plt.clf()

#plotting reweighting
#plt.plot(prob,prob,'--k', label=r'')
#plt.plot(p,prosp,'g', label=r'decision weight density $w(p)$')
#plt.plot(prob,reweighted,'r', lineWidth='3',label=r'directly from dists')

#
#plt.legend()
#plt.xlabel(r'PDF $p$')
#plt.ylabel(r'decision weight PDF $w(p)$')
#
#plt.savefig("./../probability_weights.pdf", bbox_inches='tight')
#plt.show()
#plt.clf()
#
##plotting CDFs
#plt.plot(cp[1],cp[1],'--k', label=r'')
#plt.axvline(x=0.5,LineStyle='--')
#plt.plot(cp[1],cq[1],'r', lineWidth='3',label=r'Gaussian w(p)')
#plt.plot(y,s(y,sigma*sigma/(sigma*sigma+sigma1*sigma1)),'b', lineWidth='3',label=r'KT1992$ w_{KT}(p)$')
##plt.axhline(y=0.)
#
#
#
#plt.legend()
#plt.xlabel(r'CDF $p$')
#plt.ylabel(r'reweighted CDF $w(p)$')
#
#plt.savefig("./../CDF_weights.pdf", bbox_inches='tight')
#plt.show()
#plt.clf()
#

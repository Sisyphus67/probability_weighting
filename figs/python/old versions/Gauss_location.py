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


SCfigure=True
if SCfigure:
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}
    plt.rc('font', **font)
    


pos_x=np.linspace(0.01,10,1000)
all_x=np.linspace(-10,10,1000)


#Gaussian
g_loc_p=0
g_loc_w=.18
g_scale_p=1
g_scale_w=1
def gauss_p(x):
    p=1/(np.sqrt(2*np.pi*(g_scale_p)))*np.exp(-np.power((x-g_loc_p),2)/(2*(g_scale_p)))
    return p

def cdf_gauss_p(x):
    cdf=quad(gauss_p,-10,x)
    return cdf

def gauss_w(x):
    w=1/(np.sqrt(2*np.pi*(g_scale_w)))*np.exp(-np.power((x-g_loc_w),2)/(2*(g_scale_w)))
    return w

def cdf_gauss_w(x):
    cdf=quad(gauss_w,-10,x)
    return cdf


#Exponential
e_scale_p=1
def exp_p(x):
    p=1/e_scale_p*np.exp(-x/e_scale_p)
    return p

def cdf_exp_p(x):
    cdf=quad(exp_p,0,x)
    return cdf

e_scale_w=1.5
def exp_w(x):
    w=1/e_scale_w*np.exp(-x/e_scale_w)
    return w

def cdf_exp_w(x):
    cdf=quad(exp_w,0,x)
    return cdf

#Student-t
s_scale_p=1
nu_p=1/s_scale_p
def stu_p(x):
    p=gamma((nu_p+1.)/2)/(np.sqrt(np.pi*nu_p)*gamma(nu_p/2)*np.power((1+x*x/nu_p),(nu_p+1)/2))
    return p

def cdf_stu_p(x):
    cdf=quad(stu_p,-100,x)
    return cdf

s_scale_w=2
nu_w=1/s_scale_w
def stu_w(x):
    p=gamma((nu_w+1.)/2)/(np.sqrt(np.pi*nu_w)*gamma(nu_w/2)*np.power((1+x*x/nu_w),(nu_w+1)/2))
    return p

def cdf_stu_w(x):
    cdf=quad(stu_w,-1000,x)
    return cdf






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

##Exponential
#i=0
#cp=np.zeros((2,pos_x.size))
#for x in pos_x:
#    cp[0][i]=x
#    cp[1][i]=cdf_exp_p(x)[0]
#    i=i+1
#
#i=0
#cw=np.zeros((2,pos_x.size))
#for x in pos_x:
#    cw[0][i]=x
#    cw[1][i]=cdf_exp_w(x)[0]
#    i=i+1

#Student-t
#i=0
#cp=np.zeros((2,all_x.size))
#for x in all_x:
#    cp[0][i]=x
#    cp[1][i]=cdf_stu_p(x)[0]
#    i=i+1
#
#i=0
#cw=np.zeros((2,all_x.size))
#for x in all_x:
#    cw[0][i]=x
#    cw[1][i]=cdf_stu_w(x)[0]
#    i=i+1

plt.plot(cp[1],cp[1],'--k', label=r'')
plt.axvline(x=0.5,LineStyle='--')
plt.plot(cp[1],cw[1],'b', lineWidth='3',label=r'')
plt.title('Location')

plt.legend()
plt.xlabel(r'CDF $p$')
plt.ylabel(r'decision weights CDF $w(p)$')

plt.savefig("./../Gauss_locationMK.pdf", bbox_inches='tight')
plt.show()


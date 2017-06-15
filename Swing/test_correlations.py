__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import u_functions as ufun
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

"""
NOTES

This is really hacky code that was put together to look at correlations between oscillations
that have different phase and frequency.

Conclusions: Using pearson/spearman correlation, phases and frequency must be close or signals
will not be correlated
"""

#Phase is rows
#Frequency is columns
phases = np.array([np.linspace(0,2*np.pi,100)]*100).T
freqs = np.array([np.linspace(1,10,100)]*100)
data = np.array(pd.read_csv("phase_freq_correlation_values.csv", header=None).values)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(phases, freqs, data)
plt.xlabel('Phases')
plt.ylabel('Freqs')
plt.show()
sys.exit()


t = np.linspace(0,2*np.pi,10000)
s = np.cos(t)


x = ufun.simple(t).signal()
print np.cov(x,x)
z = ufun.simple(t, phase=0.5*np.pi).signal()

#plt.plot(t,x,t,z)
#plt.show()
#plt.plot(x,z)
#plt.show()
phases = np.linspace(0,2*np.pi,100)
freqs = np.linspace(1,10,100)
pearsons = np.matrix(np.zeros([len(phases),len(freqs)]))
spearmans = np.matrix(np.zeros([len(phases),len(freqs)]))
for ii in range(len(phases)):
    print ii
    for jj in range(len(freqs)):
        y = ufun.simple(t, phase=phases[ii],freq=freqs[jj]).signal()
        coef, p = stats.pearsonr(x,y)
        pearsons[ii,jj] = coef
        coef, p = stats.spearmanr(x,y)
        spearmans[ii,jj] = coef

np.savetxt('phase_freq_correlation_values.csv', pearsons, delimiter=',')
sys.exit()
plt.plot(phases,pearsons, phases, spearmans)
plt.show()


pearsons2 = []
spearmans2 = []

for freq in freqs:
    w = ufun.simple(t,freq=freq).signal()
    coef, p = stats.pearsonr(x,w)
    pearsons2.append(coef)
    coef, p = stats.spearmanr(x,w)
    spearmans2.append(coef)

plt.plot(freqs,pearsons2, freqs, spearmans2)
plt.show()

a1=1
b1=1
c1=0
d1=0
a2=a1
b2=b1
c2=c1
d2=d1

cov = a1*np.sin(b1*t+c1)*a2*np.sin(b2*t+c2)+a1*d2*np.sin(b1*t+c1)+d1*a2*np.sin(b2*t+c2)+d1*d2
print np.mean(cov)
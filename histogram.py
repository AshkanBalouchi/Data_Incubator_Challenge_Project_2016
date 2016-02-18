#

from __future__ import division
import numpy
from pylab import *
from scipy import optimize
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as patches
import glob
import os
import math
from scipy.optimize import curve_fit
from numpy import genfromtxt

# Plot Distribution of rates for products

my_data = genfromtxt('Products.csv', dtype=float, delimiter=',',names=True)

raw=my_data["Mean_Rate"]
modified=my_data["Trusty_Mean_Rate"]
Count=my_data["Count"]

figure(figsize=(10,8))

g=where(Count>1)

counts1,bins1,patches1=plt.hist(raw[g],color='#B4045F',bins=15,range=(0.5,5.5),label=r'Raw Rate of Products')
#counts2,bins2,patches2 = plt.hist(modified[g],color='#58D3F7',bins=15,range=(0.5,5.5),label=r'Modified Rate of Products',alpha=0.7)

plt.xlim((0.5,5.5))

#bincenters1=0.5*(bins1[1:]+bins1[:-1])
#bincenters2=0.5*(bins2[1:]+bins2[:-1])
#binsize2=bins2[1:]-bins2[:-1]
#g1=where(counts1>0)
#g2=where(counts2>0)

#plt.plot(bincenters1[g1],counts1[g1],'^',c='#FE2E2E',markersize=8,markeredgecolor='#FE2E2E')
#plt.plot(bincenters2[g2],counts2[g2],'o',c='#5858FA',markersize=8,markeredgecolor='#5858FA')

ax=gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(15)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(15)
rc('font',family='serif')
rc('text',usetex=True)

plt.xlabel(r'Rate',fontsize=25)
plt.ylabel(r'Frequency',fontsize=25)
plt.suptitle('Histogram of Raw Rate of Products with More than One Review\n Amazon Musical Instruments Products Review Datasets',fontsize=20)

l=legend(bbox_to_anchor=(0.02,0.98),loc=2,prop={'size':25},borderaxespad=0,numpoints=1,fancybox=True,handletextpad=1,handlelength=1,labelspacing=0.1)
l.get_frame().set_alpha(0.5)
#ax.set_yscale('log')
plt.grid()
plt.savefig("raw_products_more_one_hist.png")
plt.show()





# Plot Histogram of Trustfullness
"""
my_data = genfromtxt('people_rate.csv', dtype=float, delimiter=',',names=True)
trust_factor=my_data["trust_factor"]
nreview=my_data["Number_of_Reviews"]
g=where(nreview>1)
figure(figsize=(10,8))

print len(nreview),len(nreview[g])

plt.hist(trust_factor[g],color='#58D3F7',bins=20)

ax=gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(17)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(17)
rc('font',family='serif')
rc('text',usetex=True)

plt.xlabel(r'Trust Factor',fontsize=25)
plt.ylabel(r'Frequency',fontsize=25)
plt.suptitle('Histogram of Trust Factor (for reviewers with more than one review)\n Amazon Musical Instruments Products Review Datasets',fontsize=20)

#l=legend(bbox_to_anchor=(0.02,0.98),loc=2,prop={'size':25},borderaxespad=0,numpoints=1,fancybox=True,handletextpad=1,handlelength=1,labelspacing=0.1)
plt.grid()
plt.savefig("trust_hist_nreviews1.png")
plt.show()
"""
# Plot Distribution of Rates
"""
my_data = genfromtxt('Review_Musical_Instruments_rates.csv', dtype=float, delimiter=',',names=True)

raw=my_data["overall"]
modified=my_data["trusty_overall"]

figure(figsize=(10,8))

counts1,bins1,patches1=plt.hist(raw,color='#B4045F',bins=15,range=(0.5,5.5),label=r'Raw Rate')
counts2,bins2,patches2=plt.hist(modified,color='#58D3F7',bins=15,range=(0.5,5.5),label=r'Modified Rate',alpha=0.7)

plt.xlim((0.5,5.5))

bincenters1=0.5*(bins1[1:]+bins1[:-1])
bincenters2=0.5*(bins2[1:]+bins2[:-1])
binsize2=bins2[1:]-bins2[:-1]
g1=where(counts1>0)
g2=where(counts2>0)

#plt.plot(bincenters1[g1],counts1[g1],'^',c='#FE2E2E',markersize=8,markeredgecolor='#FE2E2E')
#plt.plot(bincenters2[g2],counts2[g2],'o',c='#5858FA',markersize=8,markeredgecolor='#5858FA')

ax=gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(15)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(15)
rc('font',family='serif')
rc('text',usetex=True)

plt.xlabel(r'Rate',fontsize=25)
plt.ylabel(r'Frequency',fontsize=25)
plt.suptitle('Histograms of Raw Rate and Modified Rate\n Amazon Musical Instruments Products Review Datasets',fontsize=20)

l=legend(bbox_to_anchor=(0.02,0.98),loc=2,prop={'size':25},borderaxespad=0,numpoints=1,fancybox=True,handletextpad=1,handlelength=1,labelspacing=0.1)
l.get_frame().set_alpha(0.5)
#ax.set_yscale('log')
plt.grid()
plt.savefig("raw_modified_hist.png")
plt.show()
"""
# Plot Rate vs. Number of Reviews
"""
my_data = genfromtxt('counts_info.csv', dtype=float, delimiter=',',names=True)

count=my_data["Count"]

abs_m_ch=my_data["abs_Mean_Change"]
yerr_abs=my_data["abs_Mean_STD"]

abs_m_ratio_ch=my_data["abs_Mean_Ratio_Change"]
yerr_abs_ratio=my_data["abs_Mean_Ratio_STD"]

m_ch=my_data["Mean_Change"]
yerr=my_data["Mean_STD"]

m_ratio_ch=my_data["Mean_Ratio_Change"]
yerr_ratio=my_data["Mean_Ratio_STD"]

entries=my_data["Entries"]

g=where(entries>19)

#plt.plot(count[g],abs_m_ratio_ch[g],'o',c='#5858FA',markersize=8,markeredgecolor='#5858FA',alpha=0.5)
#plt.errorbar(count[g],abs_m_ch[g],yerr=yerr_abs[g],fmt='o',c='#58D3F7',markersize=8)

plt.errorbar(count[g],abs_m_ch[g],yerr=yerr_abs[g],fmt='o',c='#58D3F7',markersize=8)

plt.xlabel(r'Number of Reviews',fontsize=20)
#plt.ylabel(r'$<\Delta$Rate$ >$',fontsize=20)
plt.ylabel(r'$<|\Delta$Rate$|>$',fontsize=20)
#plt.ylabel(r'$<|\Delta$Rate/Initial Rate$|>$',fontsize=20)
plt.suptitle('Expected Value of ``Modified Rate - Raw Rate" vs. Number of Reviews\n Amazon Musical Instruments Products Review Datasets',fontsize=15)
#plt.xlim((0.9,10))

def expfunc(x,a,b,c):
    return a+b*numpy.exp(c*x)

popt, pcov = curve_fit(expfunc,count[g], abs_m_ch[g], p0=(0.15,1,-1))
yFIT = expfunc(count[g], *popt)

plt.plot(count[g], yFIT,ls='--',c='#B4045F',lw=2,label=('$a$=%s\n$b$=%s\n$c$=%s'%(tuple(numpy.round(popt,2)))))
plt.annotate(r'f = a + b exp(cx)',
             xy=(0.7,0.85),xycoords='figure fraction',color='black',
             xytext=(0.2, 0.6), textcoords='offset points',
             bbox=dict(boxstyle="round",edgecolor='black', fc='none')
             )

ax=gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(15)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(15)
rc('font',family='serif')
rc('text',usetex=True)
#ax.set_yscale('log')

l=legend(bbox_to_anchor=(0.96,0.9),loc=1,borderaxespad=0,numpoints=1,fancybox=True,handletextpad=1,handlelength=2)
plt.savefig("/Users/Azadeh/Desktop/Ashkan-Data-Incubator/count_abs-m-ch.png")
plt.show()
"""

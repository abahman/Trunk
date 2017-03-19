'''
Created on March 15, 2017

@author: ammarbahman

Note: this file plots the effectivness in 60K ECU for IJR paper

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
from scipy import exp
import scipy
import scipy.ndimage as ndimage
#import matplotlib.mlab as mlab
#import matplotlib.cm as cm
#from pylab import contourf, clf
import matplotlib as mpl
mpl.style.use('classic')

#===============================================================================
# Latex render
#===============================================================================
import matplotlib as mpl
from numpy import integer
from numba.targets.randomimpl import f_impl
#mpl.use('pgf')

def figsize(scale):
    fig_width_pt = 469.755                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
"text.usetex": True,                # use LaTeX to write all text
"font.family": "serif",
"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
"font.sans-serif": [],
"font.monospace": [],
"axes.labelsize": 10,               # LaTeX default is 10pt font.
"font.size": 10,
"legend.fontsize": 8,               # Make the legend/label fonts a little smaller
"legend.labelspacing":0.2,
"xtick.labelsize": 8,
"ytick.labelsize": 8,
"figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
"pgf.preamble": [
r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)
#===============================================================================
# END of Latex render
#===============================================================================
 

#import the excel file
df = pd.read_excel("effectivness.xlsx",sheetname='new')

Baseline = np.array(df[:]["Baseline"])
Modified = np.array(df[:]["Modified"])
Interleaved = np.array(df[:]["Interleaved"])
    
plt.plot(np.arange(1,9,1),Baseline,'--ko',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Baseline')
plt.plot(np.arange(1,9,1),Modified,'--bs',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Modified')
plt.plot(np.arange(1,9,1),Interleaved,'--r^',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Interleaved')
# plt.plot(np.arange(1,9,1),Q,'g^',markersize=4,label=r'$\dot Q$')
# 
plt.ylim(0,1)
plt.xlim(0,9)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
           [r'', r'C', r'B', r'6',r'5', r'4/A', r'3', r'2', r'1', r''])
plt.xlabel(r'Test condition')
plt.ylabel(r'$\varepsilon$ [-]')
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout() 
#plt.savefig('effectivness.pdf')
#plt.show()
plt.close()


#same plot but in bars
#ax = plt.subplot()    
plt.bar(np.arange(1,9,1)-0.2,Baseline,width=0.2,color='b',linewidth=0.9,align='center',alpha=0.9,label=r'Baseline')#hatch=5*'\\',
plt.errorbar(np.arange(1,9,1)-0.2,Baseline,yerr=0.1234*Baseline,capsize=2,elinewidth=0.7,fmt='',linestyle="None",color='k')

plt.bar(np.arange(1,9,1),Modified,width=0.2,color='g',linewidth=0.9,align='center',alpha=0.9,label=r'Modified')#hatch=4*'-'
plt.errorbar(np.arange(1,9,1),Modified,yerr=0.1234*Modified,capsize=2,elinewidth=0.7,fmt='',linestyle="None",color='k')

plt.bar(np.arange(1,9,1)+0.2,Interleaved,width=0.2,color='r',linewidth=0.9,align='center',alpha=0.9,label=r'Interleaved')#hatch=4*'x',
plt.errorbar(np.arange(1,9,1)+0.2,Interleaved,yerr=0.1234*Interleaved,capsize=2,elinewidth=0.7,fmt='',linestyle="None",color='k')

plt.ylim(0,1)
plt.xlim(0,9)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
           [r'', r'C', r'B', r'6',r'5', r'4/A', r'3', r'2', r'1', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'Test condition')
plt.ylabel(r'$\varepsilon$ [-]')
leg = plt.legend(loc='best',fancybox=False)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout() 
plt.savefig('effectivness_bar_errorbar.pdf')
plt.show()

 

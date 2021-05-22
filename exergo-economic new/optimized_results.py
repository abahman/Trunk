import os,sys
import numpy as np
import math
from math import log, pi
import pylab
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.mlab as ml
import pandas as pd
mpl.style.use('classic')
mpl.style.use('Elsevier.mplstyle')
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['figure.figsize'] = [6,4]
mpl.rcParams['legend.labelspacing'] = 0.2
mpl.rcParams['legend.numpoints'] = 1

# #===============================================================================
# # Latex render
# #===============================================================================
# #mpl.use('pgf')
#  
# def figsize(scale):
#     fig_width_pt = 469.755                          # Get this from LaTeX using \the\textwidth
#     inches_per_pt = 1.0/72.27                       # Convert pt to inch
#     golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
#     fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
#     fig_height = fig_width*golden_mean              # height in inches
#     fig_size = [fig_width,fig_height]
#     return fig_size
#  
# pgf_with_latex = {                      # setup matplotlib to use latex for output
# "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
# "text.usetex": True,                # use LaTeX to write all text
# "font.family": "serif",
# "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
# "font.sans-serif": [],
# "font.monospace": [],
# "axes.labelsize": 10,               # LaTeX default is 10pt font.
# "font.size": 10,
# "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
# "legend.labelspacing":0.2,
# "xtick.labelsize": 8,
# "ytick.labelsize": 8,
# "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
# "pgf.preamble": [
# r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
# r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
#         ]
#     }
# mpl.rcParams.update(pgf_with_latex)
# #===============================================================================
# # END of Latex render
# #===============================================================================
    
def savefigs(name):
    #plt.savefig(name+'.eps')
    plt.savefig('images/'+name+'.pdf')
    plt.savefig(name+'.png',dpi=600)
    #plt.show()


#########################
##### optimized results (COP)  #######
#########################  
#import data from excel file
df = pd.read_excel('bar.xlsx',sheet_name='COP',header=0) #file name 
       
y1 = df['A']
y2 = df['B']
y3 = df['AB']
 
plt.bar(np.arange(1,6,1)-0.2,y1,width=0.2,color='yellow',linewidth=0.9,align='center',alpha=0.9,label=r'Point A',hatch=5*'\\')    
    
plt.bar(np.arange(1,6,1),y2,width=0.2,color='orange',linewidth=0.9,align='center',alpha=0.9,label=r'Point B',hatch=2*'x')  
        
plt.bar(np.arange(1,6,1)+0.2,y3,width=0.2,color='brown',linewidth=0.9,align='center',alpha=0.9,label=r'Point AB',hatch=2*'//') 
 
# plt.ylim(0,6)
# plt.xlim(0,7)
plt.xticks([0, 1, 2, 3, 4, 5, 6],
           [r'', r'R-32', r'R-290', r'R-410A',r'R-454A', r'R-452B', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
# plt.xlabel(r'Test condition')
plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]')
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('optimized_results_COP')
plt.show()
plt.close()



#########################
#####  optimized results (UCH) #######
#########################  
#import data from excel file
df = pd.read_excel('bar.xlsx',sheet_name='UCH',header=0) #file name 
       
y1 = df['A']
y2 = df['B']
y3 = df['AB']
 
plt.bar(np.arange(1,6,1)-0.2,y1,width=0.2,color='m',linewidth=0.9,align='center',alpha=0.9,label=r'Point A',hatch=5*'\\')    
    
plt.bar(np.arange(1,6,1),y2,width=0.2,color='c',linewidth=0.9,align='center',alpha=0.9,label=r'Point B',hatch=2*'x')  
        
plt.bar(np.arange(1,6,1)+0.2,y3,width=0.2,color='r',linewidth=0.9,align='center',alpha=0.9,label=r'Point AB',hatch=2*'//') 
 
plt.ylim(0,0.3)
# plt.xlim(0,7)
plt.xticks([0, 1, 2, 3, 4, 5, 6],
           [r'', r'R-32', r'R-290', r'R-410A',r'R-454A', r'R-452B', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are on
    top=False,         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
# plt.xlabel(r'Test condition')
plt.ylabel(r'UCH [$\$$ kWh$^{-1}$]')
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('optimized_results_UCH')
plt.show()
plt.close()
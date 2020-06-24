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
    plt.savefig(name+'.pdf')
#     plt.savefig(name+'.png',dpi=600)
    #plt.show()
    
#import data from excel file
df1 = pd.read_excel('data_tables.xlsx',sheet_name='60K ECU',header=0) #file name
df2 = pd.read_excel('data_tables.xlsx',sheet_name='36K ECU',header=0) #file name
df3 = pd.read_excel('data_tables.xlsx',sheet_name='18K ECU',header=0) #file name
#assign axes
x1 = df1['Q_dot_evap'][1:7]
y1 = df1['Q_dot_evap_airside'][1:7]
x2 = df2['Q_dot_evap'][1:7]
y2 = df2['Q_dot_evap_airside'][1:7]
x3 = df3['Q_dot_evap'][1:7]
y3 = df3['Q_dot_evap_airside'][1:7]

#########################
##### parity_1 #######
#########################
s = 40  # size of points

fig, ax = plt.subplots(figsize=(4,4))
im = ax.scatter(x1, y1, c='r', s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =1.0,label='5-RT ECU')
im = ax.scatter(x2, y2, c='b', s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =1.0,label='3-RT ECU')
im = ax.scatter(x3, y3, c='g', s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =1.0,label='1.5-RT ECU')

#error axes
w=0.06 #Error
ax_min = 0
ax_max = 25 #x and y-axes max scale tick
upp_txt = (ax_min+ax_max) / 2.2 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = (ax_min+ax_max) / 2.0 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'$-${:0.0f}$\%$'.format(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}$\%$'.format(w*100),ha='right',va='bottom')
leg=ax.legend(loc='upper left',scatterpoints=1,scatteryoffsets=[0.5])
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
ax.set_xlim((ax_min,ax_max))
ax.set_ylim((ax_min,ax_max))
plt.xlabel(r'$\dot Q_{evap,r}$ (kW)')
plt.ylabel(r'$\dot Q_{evap,a}$ (kW)')
plt.tight_layout(pad=0.2)       
plt.savefig('parity_1.pdf')
plt.show()
plt.close()
     
#########################
##### parity_2 #######
#########################
fig=pylab.figure(figsize=(4,4))
plt.plot(x1,y1,'rs',ms = 6,mec='black',mew=0.5,label='5-RT ECU')
plt.plot(x2,y2,'b^',ms = 6,mec='black',mew=0.5,label='3-RT ECU')
plt.plot(x3,y3,'go',ms = 6,mec='black',mew=0.5,label='1.5-RT ECU')
plt.xlabel(r'$\dot Q_{evap,r}$ (kW)')
plt.ylabel(r'$\dot Q_{evap,a}$ (kW)')
Tmin = 0
Tmax = 25
x=[Tmin,Tmax]
y=[Tmin,Tmax]
y105=[1.06*Tmin,1.06*Tmax]
y95=[0.94*Tmin,0.94*Tmax]
plt.plot(x,y,'k-')
plt.fill_between(x,y105,y95,color='black',alpha=0.2)    
plt.xlim(Tmin,Tmax)
plt.ylim(Tmin,Tmax)
leg=plt.legend(loc='upper left',scatterpoints=1,scatteryoffsets=[0.5])
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2)        
plt.tick_params(direction='in')
savefigs('parity_2')
plt.show()
plt.close()


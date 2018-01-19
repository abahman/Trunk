'''
Created on Jan 16, 2018

@author: ammarbahman

Note: this file plots the percentage in VI ECU for IJR paper

'''
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('classic')
mpl.style.use('Elsevier.mplstyle')
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['figure.figsize'] = [6,4]

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
 

#import the excel file
df = pd.read_excel("VIpercentage.xlsx")
COP_7K = np.array(df[:]["COP_7K"])
Q_7K = np.array(df[:]["Q_7K"])
Tdis_7K = np.array(df[:]["Tdis_7K"])
COP_0K = np.array(df[:]["COP_0K"])
Q_0K = np.array(df[:]["Q_0K"])
Tdis_0K = np.array(df[:]["Tdis_0K"])

#flip the array for better plotting
COP_7K = np.flip(COP_7K,0)
Q_7K = np.flip(Q_7K,0)
Tdis_7K = np.flip(Tdis_7K,0)
COP_0K = np.flip(COP_0K,0)
Q_0K = np.flip(Q_0K,0)
Tdis_0K = np.flip(Tdis_0K,0)

#ploting COP
plt.bar(np.arange(1,9,1)-0.1,COP_7K,width=0.2,color='r',linewidth=0.9,align='center',alpha=0.9,label=r'Superheated',hatch=5*'\\')    
plt.errorbar(np.arange(1,9,1)-0.1,COP_7K,yerr=0.0277*COP_7K,fmt='',linestyle="None",color='k')

plt.bar(np.arange(1,9,1)+0.1,COP_0K,width=0.2,color='b',linewidth=0.9,align='center',alpha=0.9,label=r'Saturated',hatch=2*'//')
plt.errorbar(np.arange(1,9,1)+0.1,COP_0K,yerr=0.0277*COP_0K,fmt='',linestyle="None",color='k')

plt.axhline(y=0, color='k') #draw a black at y=0

plt.ylim(-4,6)
plt.xlim(0,9)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
           [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are on
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'Test condition')
plt.ylabel(r'$\mathrm{COP}_{sys}$ Improvement [%]')
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout() 
plt.savefig('VIpercentage_COP.pdf')
plt.show()


#ploting Capacity
plt.bar(np.arange(1,9,1)-0.1,Q_7K,width=0.2,color='yellow',linewidth=0.9,align='center',alpha=0.9,label=r'Superheated',hatch=5*'\\')    
plt.errorbar(np.arange(1,9,1)-0.1,Q_7K,yerr=0.0091*Q_7K,fmt='',linestyle="None",color='k')

plt.bar(np.arange(1,9,1)+0.1,Q_0K,width=0.2,color='brown',linewidth=0.9,align='center',alpha=0.9,label=r'Saturated',hatch=2*'//')
plt.errorbar(np.arange(1,9,1)+0.1,Q_0K,yerr=0.0091*Q_0K,fmt='',linestyle="None",color='k')

#plt.axhline(y=0, color='k') #draw a black at y=0

plt.ylim(0,20)
plt.xlim(0,9)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
           [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'Test condition')
plt.ylabel(r'$\dot Q_{evap}$ Improvement [%]')
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout() 
plt.savefig('VIpercentage_capacity.pdf')
plt.show()


#ploting Discharge temperature
#plt.bar(np.arange(1,9,1)-0.1,Tdis_7K,width=0.2,color='orange',linewidth=0.9,align='center',alpha=0.9,label=r'Superheated',hatch=5*'\\')
plt.plot(np.arange(1,9,1),Tdis_7K,'^-',color='orange',linewidth=0.9,label=r'Superheated')    
plt.errorbar(np.arange(1,9,1),Tdis_7K,yerr=1.1,fmt='',linestyle="None",color='k')

#plt.bar(np.arange(1,9,1)+0.1,Tdis_0K,width=0.2,color='green',linewidth=0.9,align='center',alpha=0.9,label=r'Saturated',hatch=2*'//')
plt.plot(np.arange(1,9,1),Tdis_0K,'s-',color='green',linewidth=0.9,label=r'Saturated')
plt.errorbar(np.arange(1,9,1),Tdis_0K,yerr=1.1,fmt='',linestyle="None",color='k')

#plt.axhline(y=0, color='k') #draw a black at y=0

plt.ylim(-10,10)
plt.xlim(0,9)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
           [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are on
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'Test condition')
plt.ylabel(r'$T_{dis}$ Improvement [$\degree$C]') #{\textdegree}C
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout() 
plt.savefig('VIpercentage_dischargeTemp.pdf')
plt.show()
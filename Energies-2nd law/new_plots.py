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
df = pd.read_excel('data_tables.xlsx',sheet_name='new_plots',header=0) #file name
COP_18K = np.array(df[:]["COP_18K"])
COP_36 = np.array(df[:]["COP_36"])
COP_60 = np.array(df[:]["COP_60"])
epsilon_18 = np.array(df[:]["epsilon_18"])*100
epsilon_36 = np.array(df[:]["epsilon_36"])*100
epsilon_60 = np.array(df[:]["epsilon_60"])*100
I_18 = np.array(df[:]["I_18"])*1000
I_36 = np.array(df[:]["I_36"])*1000
I_60 = np.array(df[:]["I_60"])*1000

#flip the array for better plotting
COP_18K = np.flip(COP_18K,0)
COP_36 = np.flip(COP_36,0)
COP_60 = np.flip(COP_60,0)
epsilon_18 = np.flip(epsilon_18,0)
epsilon_36 = np.flip(epsilon_36,0)
epsilon_60 = np.flip(epsilon_60,0)
I_18 = np.flip(I_18,0)
I_36 = np.flip(I_36,0)
I_60 = np.flip(I_60,0)

# #ploting COP
plt.bar(np.arange(1,7,1)-0.2,COP_60,width=0.2,color='r',linewidth=0.9,align='center',alpha=0.9,label=r'5-RT ECU',hatch=5*'\\')    
plt.errorbar(np.arange(1,7,1)-0.2,COP_60,yerr=0.061,fmt='',linestyle="None",color='k')

plt.bar(np.arange(1,7,1),COP_36,width=0.2,color='b',linewidth=0.9,align='center',alpha=0.9,label=r'3-RT ECU',hatch=2*'x')
plt.errorbar(np.arange(1,7,1),COP_36,yerr=0.078,fmt='',linestyle="None",color='k')
 
plt.bar(np.arange(1,7,1)+0.2,COP_18K,width=0.2,color='g',linewidth=0.9,align='center',alpha=0.9,label=r'1.5-RT ECU',hatch=2*'//')
plt.errorbar(np.arange(1,7,1)+0.2,COP_18K,yerr=0.036,fmt='',linestyle="None",color='k')
 
# plt.axhline(y=0, color='k') #draw a black at y=0
 
plt.ylim(0,6)
plt.xlim(0,7)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7],
           [r'', r'1', r'2', r'3',r'4', r'5', r'6', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are on
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'Test condition')
plt.ylabel(r'$\mathrm{COP}_{c}$ ($-$)')
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
plt.savefig('COP.pdf')
plt.show()


# #ploting irreversibility
plt.bar(np.arange(1,7,1)-0.2,I_60,width=0.2,color='yellow',linewidth=0.9,align='center',alpha=0.9,label=r'5-RT ECU',hatch=5*'\\')    
plt.errorbar(np.arange(1,7,1)-0.2,I_60,yerr=165.9,fmt='',linestyle="None",color='k')

plt.bar(np.arange(1,7,1),I_36,width=0.2,color='orange',linewidth=0.9,align='center',alpha=0.9,label=r'3-RT ECU',hatch=2*'x')
plt.errorbar(np.arange(1,7,1),I_36,yerr=135.9,fmt='',linestyle="None",color='k')

plt.bar(np.arange(1,7,1)+0.2,I_18,width=0.2,color='brown',linewidth=0.9,align='center',alpha=0.9,label=r'1.5-RT ECU',hatch=2*'//')
plt.errorbar(np.arange(1,7,1)+0.2,I_18,yerr=11.0,fmt='',linestyle="None",color='k')
 
#plt.axhline(y=0, color='k') #draw a black at y=0
 
plt.ylim(0,9000)
plt.xlim(0,7)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7],
           [r'', r'1', r'2', r'3',r'4', r'5', r'6', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'Test condition')
plt.ylabel(r'$\dot I$ (W)')
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
plt.savefig('irrev_all.pdf')
plt.show()


#ploting epsilon
plt.bar(np.arange(1,7,1)-0.2,epsilon_60,width=0.2,color='m',linewidth=0.9,align='center',alpha=0.9,label=r'5-RT ECU',hatch=5*'\\')    
plt.errorbar(np.arange(1,7,1)-0.2,epsilon_60,yerr=1.13,fmt='',linestyle="None",color='k')

plt.bar(np.arange(1,7,1),epsilon_36,width=0.2,color='c',linewidth=0.9,align='center',alpha=0.9,label=r'3-RT ECU',hatch=2*'x')
plt.errorbar(np.arange(1,7,1),epsilon_36,yerr=1.24,fmt='',linestyle="None",color='k')

plt.bar(np.arange(1,7,1)+0.2,epsilon_18,width=0.2,color='r',linewidth=0.9,align='center',alpha=0.9,label=r'1.5-RT ECU',hatch=2*'//')
plt.errorbar(np.arange(1,7,1)+0.2,epsilon_18,yerr=0.23,fmt='',linestyle="None",color='k')

# plt.plot(np.arange(1,7,1),epsilon_60,'^-',color='orange',linewidth=0.9,label=r'5-RT ECU')    
# plt.errorbar(np.arange(1,7,1),epsilon_60,yerr=1.13,fmt='',linestyle="None",color='k')
# 
# #plt.bar(np.arange(1,9,1)+0.1,Tdis_0K,width=0.2,color='green',linewidth=0.9,align='center',alpha=0.9,label=r'Saturated',hatch=2*'//')
# plt.plot(np.arange(1,7,1),epsilon_36,'s-',color='green',linewidth=0.9,label=r'3-RT ECU')
# plt.errorbar(np.arange(1,7,1),epsilon_36,yerr=1.24,fmt='',linestyle="None",color='k')
# 
# plt.plot(np.arange(1,7,1),epsilon_18,'o-',color='red',linewidth=0.9,label=r'1.5-RT ECU')
# plt.errorbar(np.arange(1,7,1),epsilon_18,yerr=0.23,fmt='',linestyle="None",color='k')

#plt.axhline(y=0, color='k') #draw a black at y=0

plt.ylim(0,20)
plt.xlim(0,7)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7],
           [r'', r'1', r'2', r'3',r'4', r'5', r'6', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are on
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'Test condition')
plt.ylabel(r'$\varepsilon_{c}$ (%)') 
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
plt.savefig('epsilon.pdf')
plt.show()
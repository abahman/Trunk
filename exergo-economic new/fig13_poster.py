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
    #plt.savefig(name+'.png',dpi=600)
    #plt.show()


#########################
##### Figure 13 plot #######
#########################  
#import data from excel file
df = pd.read_excel('bar.xlsx',sheet_name='Sheet3',header=0) #file name 
       
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
       
fig, host = plt.subplots(figsize=(9,4))
fig.subplots_adjust(right=0.7)
       
par1 = host.twinx()
par2 = host.twinx()
       
par1.spines["left"].set_position(("axes", -0.18)) # green one
par2.spines["left"].set_position(("axes", -0.35)) # red one
make_patch_spines_invisible(par1)
make_patch_spines_invisible(par2)
par1.spines["left"].set_visible(True)
par1.yaxis.set_label_position('left')
par1.yaxis.set_ticks_position('left')
par2.spines["left"].set_visible(True)
par2.yaxis.set_label_position('left')
par2.yaxis.set_ticks_position('left')
       
y1 = df['COP']
y2 = df['eta']
y3 = df['UCC']
 
host.bar(np.arange(1,6,1)-0.2,y1,width=0.2,color='b',linewidth=0.9,align='center',alpha=0.9,label=r'$\mathrm{COP_{C}}$',hatch=5*'\\')    
    
par1.bar(np.arange(1,6,1),y2,width=0.2,color='g',linewidth=0.9,align='center',alpha=0.9,label=r'$\eta_{ex}$',hatch=2*'x')  
        
par2.bar(np.arange(1,6,1)+0.2,y3,width=0.2,color='r',linewidth=0.9,align='center',alpha=0.9,label=r'UCC',hatch=2*'//') 
 
# host.set_xlim(2,20)
host.set_ylim(0, 4)
par1.set_ylim(0, 60)
par2.set_ylim(0, 0.8)
plt.xticks([0, 1, 2, 3, 4, 5, 6],
           [r'', r'R-32', r'R-290', r'R-410A',r'R-454A', r'R-452B', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are on
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
 


host.xaxis.set_visible(False)


# host.set_xlabel(r'$Sh_{econ}$ [$\degree$C]')
host.set_ylabel(r'$\mathrm{COP_{C}}$ [$-$]')
par1.set_ylabel(r'$\eta_{ex}$ [%]')
par2.set_ylabel(r'UCC [$\$$/kWh]')
       
host.yaxis.label.set_color('blue')
par1.yaxis.label.set_color('green')
par2.yaxis.label.set_color('red')
       
tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', colors='blue', **tkw)
par1.tick_params(axis='y', colors='green', **tkw)
par2.tick_params(axis='y', colors='red', **tkw)
host.tick_params(axis='x', **tkw)
       
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
       
legend_elements = [Patch(facecolor='blue', edgecolor='k',hatch=5*'\\',label='$\mathrm{COP_{C}}$'),
                   Patch(facecolor='green', edgecolor='k',hatch=3*'x',label='$\eta_{ex}$'),
                   Patch(facecolor='red', edgecolor='k',hatch=2*'//',label='UCC'),
                   ]
 
       
leg = host.legend(handles=legend_elements,fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig13_plot_poster')
plt.show()
plt.close()



#########################
##### Figure 13table  #######
######################### 
cell_text = [[13.56,13.89,13.434,13.83,13.16],
[64.06,64.02,64.006,64.10,64.13],
[5.04,7.46,7.24,6.73,4.86],
[7.94,7.77,7.95,7.96,7.67],
[6.74,7.17,7.35,7.64,6.13],
[29.36,52.57,35.76,41.99,34.00],
[13.36,13.62,13.26,13.80,13.34],
[13.36,13.62,13.26,13.80,13.34],
[1.16,1.35,1.57,1.10,1.43],
[0.001,0.00095,0.001,0.001,0.0009],
[0.001,0.0011,0.001,0.0008,0.0005],
[1.87,1.84,1.94,1.77,1.90],
[2.37,2.13,2.40,2.03,2.36],
[0.18,0.17,0.18,0.12,0.19],
[3.15,3.43,3.06,3.62,3.13],
[42.15,41.61,42.22,32.12,44.94],
[0.65,0.68,0.70,0.66,0.68]]


rows = [r'$T_{evap}$ [$\degree$C]',
r'$T_{cond}$ [$\degree$C]',
r'$Sh_{econ}$ [$\degree$C]',
r'$Sc_{cond}$ [$\degree$C]',
r'$Sh_{evap}$ [$\degree$C]',
r'$\dot Q_{InHX}$ [kW]',
r'$\dot Q_{OutHX}$ [kW]',
r'$\dot Q_{evap}$  [kW]',
r'$\dot Q_{econ}$ [kW]',
r'$\dot W_{pump,1}$ [kW]',
r'$\dot W_{pump,2}$ [kW]',
r'$\dot W_{comp,1}$ [kW]',
r'$\dot W_{comp,2}$ [kW]',
r'$\dot{Ex}_{P}^{OutHX}$ [kW]',
r'$\mathrm{COP_{C}}$ [$-$]',
r'$\eta_{ex}$ [%]',
r'UCC [$\$$/kWh]']



col_labels = [r'R-32', r'R-290', r'R-410A',r'R-454A', r'R-452B']
row_labels = rows
table_vals = cell_text

# Draw table
the_table = plt.table(cellText=table_vals,
                      colWidths=[0.15] * len(rows),cellLoc = 'center',
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      loc='center',edges='open')
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(1.2, 1.2)
plt.subplots_adjust(left=0.3)

# Removing ticks and spines enables you to get the figure only with table
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
for pos in ['right','top','bottom','left']:
    plt.gca().spines[pos].set_visible(False)
    
savefigs('fig13_table_poster')
plt.show()
plt.close()
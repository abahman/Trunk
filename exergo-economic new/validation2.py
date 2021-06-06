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
##### validation plot #######
#########################  
#import data from excel file
df = pd.read_excel('validation.xlsx',sheet_name='Bertsch',header=0) #file name 
df2 = pd.read_excel('validation.xlsx',sheet_name='Cao',header=0) #file name 
       
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
       
fig, host = plt.subplots(figsize=(6,4))
# fig.subplots_adjust(right=0.8)
       
par1 = host.twinx()
par2 = host.twiny()

par3 = par1.twiny() 
# par1.spines["left"].set_position(("axes", -0.15)) # green one
# make_patch_spines_invisible(par1)
# par1.spines["left"].set_visible(True)
# par1.yaxis.set_label_position('left')
# par1.yaxis.set_ticks_position('left')



x1 = df['T_eva'] 
y1 = df['COP_Bertsch']
y2 = df['COP_model']

x2 = df2['T_cond']
y3 = df2['COP_Cao']
y4 = df2['COP_model']

y5 = df2['eta_Cao']
y6 = df2['eta_model']

par2.plot(x1,y1,'ro',label=r'Bertsch et al. (2008) ($\mathrm{COP_{H}}$)')
par2.plot(x1,y2,'r-',label=r'Present study ($\mathrm{COP_{H}}$)') 

host.plot(x2,y3,'bs',label=r'Cao et al. (2014) ($\mathrm{COP_{H}}$)') 
host.plot(x2,y4,'b-',label=r'Present study ($\mathrm{COP_{H}}$)')

par1.plot(x2,y5,'g^',label=r'Cao et al. (2014) ($\eta_{ex}$)') 
par1.plot(x2,y6,'g-',label=r'Present study ($\eta_{ex}$)')
    
# par1.bar(np.arange(1,6,1),y2,width=0.2,color='g',linewidth=0.9,align='center',alpha=0.9,label=r'$\eta_{ex}$',hatch=2*'x')  
        
 
host.set_ylim(1.9, 3.9)
host.set_xlim(96, 110)
par3.set_xlim(-35, -5)
par1.set_ylim(39, 49)

# plt.xticks([0, 1, 2, 3, 4, 5, 6],
#            [r'', r'R-32', r'R-290', r'R-410A',r'R-454A', r'R-452B', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='off',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
 


# host.xaxis.set_visible(False)
host.annotate('', xy=(99,2.8), xytext=(99,3.27), arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0))
host.annotate('', xy=(98,2.6), xytext=(98,2.13), arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0))

              
par2.set_xlabel(r'$T_{evap}$ [$\degree$C]')
host.set_ylabel(r'$\mathrm{COP_{H}}$ [$-$]')
par1.set_ylabel(r'$\eta_{ex}$ [%]')
host.set_xlabel(r'$T_{cond}$ [$\degree$C]')
       
# host.yaxis.label.set_color('blue')
# par1.yaxis.label.set_color('green')
       
# tkw = dict(size=4, width=1.5)
# host.tick_params(axis='y', colors='blue', **tkw)
# par1.tick_params(axis='y', colors='green', **tkw)
# host.tick_params(axis='x', **tkw)





from matplotlib.patches import Patch
from matplotlib.lines import Line2D
        
legend_elements = [Line2D([], [], color='red', marker='o',linestyle='',label='$\mathrm{COP_{H}}$ (Bertsch et al., 2008)'),
                   Line2D([], [], color='blue', marker='s',linestyle='',label='$\mathrm{COP_{H}}$ (Cao et al., 2014)'),
                   Line2D([], [], color='green', marker='^',linestyle='',label='$\eta_{ex}$ (Cao et al., 2014)'),
                   Line2D([], [], color='black', marker='', linestyle='-',label='Present model'),
                   ]
  
# leg = host.legend(handles=legend_elements,fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))        
leg = host.legend(handles=legend_elements,fancybox=False,numpoints=1,loc='best')
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('validation2')
plt.show()
plt.close()
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
    #plt.savefig(name+'.png',dpi=600)
    #plt.show()


#########################
##### Figure 4 #######
#########################
#import data from excel file
df1 = pd.read_excel('Base_R32.xlsx',sheet_name='T_evap',header=0) #file name
df2 = pd.read_excel('Base_R290.xlsx',sheet_name='T_evap',header=0) #file name
df3 = pd.read_excel('Base_R410A.xlsx',sheet_name='T_evap',header=0) #file name
df4 = pd.read_excel('Base_R454A.xlsx',sheet_name='T_evap',header=0) #file name
df5 = pd.read_excel('Base_R452B.xlsx',sheet_name='T_evap',header=0) #file name  
  
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
  
  
p1, = host.plot(df1['X']-273,df1['COP_heating'],'bo-',label=r'R-32')
host.plot(df2['X']-273,df2['COP_heating'],'bs-',label=r'R-290')  
host.plot(df3['X']-273,df3['COP_heating'],'b^-',label=r'R-410A')  
host.plot(df4['X']-273,df4['COP_heating'],'bd-',label=r'R-454A')  
host.plot(df5['X']-273,df5['COP_heating'],'P-',color='b',label=r'R-452B')
  
p2, = par1.plot(df1['X']-273,df1['eta_ex_total'],'go-',label=r'R-32')
par1.plot(df2['X']-273,df2['eta_ex_total'],'gs-',label=r'R-290')  
par1.plot(df3['X']-273,df3['eta_ex_total'],'g^-',label=r'R-410A')  
par1.plot(df4['X']-273,df4['eta_ex_total'],'gd-',label=r'R-454A')  
par1.plot(df5['X']-273,df5['eta_ex_total'],'P-',color='g',label=r'R-452B')   
  
p3, = par2.plot(df1['X']-273,df1['UCP_H'],'ro-',label=r'R-32')
par2.plot(df2['X']-273,df2['UCP_H'],'rs-',label=r'R-290')  
par2.plot(df3['X']-273,df3['UCP_H'],'r^-',label=r'R-410A')  
par2.plot(df4['X']-273,df4['UCP_H'],'rd-',label=r'R-454A')  
par2.plot(df5['X']-273,df5['UCP_H'],'P-',color='r',label=r'R-452B')  
  
  
host.set_xlim(-10, 8)
host.set_ylim(2.6, 4)
par1.set_ylim(10, 90)
par2.set_ylim(0, 0.5)
  
host.set_xlabel(r'$T_{evap}$ [$\degree$C]')
host.set_ylabel(r'$\mathrm{COP_{H}}$ [$-$]')
par1.set_ylabel(r'$\eta_{ex}$ [%]')
par2.set_ylabel(r'UCH [$\$$ kWh$^{-1}$]')
  
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
  
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='k', label='R-32'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='k', label='R-290'),
                   Line2D([0], [0], marker='^', color='w', markerfacecolor='k', label='R-410A'),
                   Line2D([0], [0], marker='d', color='w', markerfacecolor='k', label='R-454A'),
                   Line2D([0], [0], marker='P', color='w', markerfacecolor='k', label='R-452B'),
                   ]
  
  
leg = host.legend(handles=legend_elements,fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig4')
plt.show()
plt.close()
  
#########################
##### Figure 4a #######
#########################
plt.plot(df1['X']-273,df1['COP_heating'],'ro-',label=r'R-32')    
plt.plot(df2['X']-273,df2['COP_heating'],'bs-',label=r'R-290')  
plt.plot(df3['X']-273,df3['COP_heating'],'g^-',label=r'R-410A')  
plt.plot(df4['X']-273,df4['COP_heating'],'yd-',label=r'R-454A')  
plt.plot(df5['X']-273,df5['COP_heating'],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(100,600)
plt.xlim(-10,8)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$T_{evap}$ [$\degree$C]')
plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig4a')
plt.show()
plt.close()
  
#########################
##### Figure 4b #######
#########################
plt.plot(df1['X']-273,df1['eta_ex_total'],'ro-',label=r'R-32')    
plt.plot(df2['X']-273,df2['eta_ex_total'],'bs-',label=r'R-290')  
plt.plot(df3['X']-273,df3['eta_ex_total'],'g^-',label=r'R-410A')  
plt.plot(df4['X']-273,df4['eta_ex_total'],'yd-',label=r'R-454A')  
plt.plot(df5['X']-273,df5['eta_ex_total'],'P-',color='k',label=r'R-452B')   
  
    
plt.ylim(15,45)
plt.xlim(-10,8)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$T_{evap}$ [$\degree$C]')
plt.ylabel(r'$\eta_{ex}$ [%]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig4b')
plt.show()
plt.close()
  
#########################
##### Figure 4c #######
#########################
fig, ax = plt.subplots(figsize=(7.5,4))
plt.plot(df1['X']-273,df1['UCP_H'],'ro-',label=r'R-32')    
plt.plot(df2['X']-273,df2['UCP_H'],'bs-',label=r'R-290')  
plt.plot(df3['X']-273,df3['UCP_H'],'g^-',label=r'R-410A')  
plt.plot(df4['X']-273,df4['UCP_H'],'yd-',label=r'R-454A')  
plt.plot(df5['X']-273,df5['UCP_H'],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(100,600)
plt.xlim(-10,8)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$T_{evap}$ [$\degree$C]')
plt.ylabel(r'UCH [$\$$ kWh$^{-1}$]') #{\textdegree}C
leg = plt.legend(fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig4c')
plt.show()
plt.close()
 
 
#########################
##### Figure 5 #######
#########################
#import data from excel file
df1 = pd.read_excel('Base_R32.xlsx',sheet_name='T_cond',header=0) #file name
df2 = pd.read_excel('Base_R290.xlsx',sheet_name='T_cond',header=0) #file name
df3 = pd.read_excel('Base_R410A.xlsx',sheet_name='T_cond',header=0) #file name
df4 = pd.read_excel('Base_R454A.xlsx',sheet_name='T_cond',header=0) #file name
df5 = pd.read_excel('Base_R452B.xlsx',sheet_name='T_cond',header=0) #file name  
  
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
  
  
p1, = host.plot(df1['X'][0:11]-273,df1['COP_heating'][0:11],'bo-',label=r'R-32')
host.plot(df2['X'][0:11]-273,df2['COP_heating'][0:11],'bs-',label=r'R-290')  
host.plot(df3['X'][0:11]-273,df3['COP_heating'][0:11],'b^-',label=r'R-410A')  
host.plot(df4['X'][0:11]-273,df4['COP_heating'][0:11],'bd-',label=r'R-454A')  
host.plot(df5['X'][0:11]-273,df5['COP_heating'][0:11],'P-',color='b',label=r'R-452B')
  
p2, = par1.plot(df1['X'][0:11]-273,df1['eta_ex_total'][0:11],'go-',label=r'R-32')
par1.plot(df2['X'][0:11]-273,df2['eta_ex_total'][0:11],'gs-',label=r'R-290')  
par1.plot(df3['X'][0:11]-273,df3['eta_ex_total'][0:11],'g^-',label=r'R-410A')  
par1.plot(df4['X'][0:11]-273,df4['eta_ex_total'][0:11],'gd-',label=r'R-454A')  
par1.plot(df5['X'][0:11]-273,df5['eta_ex_total'][0:11],'P-',color='g',label=r'R-452B')   
  
p3, = par2.plot(df1['X'][0:11]-273,df1['UCP_H'][0:11],'ro-',label=r'R-32')
par2.plot(df2['X'][0:11]-273,df2['UCP_H'][0:11],'rs-',label=r'R-290')  
par2.plot(df3['X'][0:11]-273,df3['UCP_H'][0:11],'r^-',label=r'R-410A')  
par2.plot(df4['X'][0:11]-273,df4['UCP_H'][0:11],'rd-',label=r'R-454A')  
par2.plot(df5['X'][0:11]-273,df5['UCP_H'][0:11],'P-',color='r',label=r'R-452B')  
  
  
host.set_xlim(54, 66)
host.set_ylim(2.6, 4)
par1.set_ylim(10, 90)
par2.set_ylim(0, 0.5)
  
host.set_xlabel(r'$T_{cond}$ [$\degree$C]')
host.set_ylabel(r'$\mathrm{COP_{H}}$ [$-$]')
par1.set_ylabel(r'$\eta_{ex}$ [%]')
par2.set_ylabel(r'UCH [$\$$ kWh$^{-1}$]')
  
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
  
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='k', label='R-32'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='k', label='R-290'),
                   Line2D([0], [0], marker='^', color='w', markerfacecolor='k', label='R-410A'),
                   Line2D([0], [0], marker='d', color='w', markerfacecolor='k', label='R-454A'),
                   Line2D([0], [0], marker='P', color='w', markerfacecolor='k', label='R-452B'),
                   ]
  
  
leg = host.legend(handles=legend_elements,fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig5')
plt.show()
plt.close()
  
#########################
##### Figure 5a #######
#########################
plt.plot(df1['X'][0:11]-273,df1['COP_heating'][0:11],'ro-',label=r'R-32')    
plt.plot(df2['X'][0:11]-273,df2['COP_heating'][0:11],'bs-',label=r'R-290')  
plt.plot(df3['X'][0:11]-273,df3['COP_heating'][0:11],'g^-',label=r'R-410A')  
plt.plot(df4['X'][0:11]-273,df4['COP_heating'][0:11],'yd-',label=r'R-454A')  
plt.plot(df5['X'][0:11]-273,df5['COP_heating'][0:11],'P-',color='k',label=r'R-452B')   
  
    
plt.ylim(2.6,3.8)
plt.xlim(54, 66)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$T_{cond}$ [$\degree$C]')
plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig5a')
plt.show()
plt.close()
  
#########################
##### Figure 5b #######
#########################
plt.plot(df1['X'][0:11]-273,df1['eta_ex_total'][0:11],'ro-',label=r'R-32')    
plt.plot(df2['X'][0:11]-273,df2['eta_ex_total'][0:11],'bs-',label=r'R-290')  
plt.plot(df3['X'][0:11]-273,df3['eta_ex_total'][0:11],'g^-',label=r'R-410A')  
plt.plot(df4['X'][0:11]-273,df4['eta_ex_total'][0:11],'yd-',label=r'R-454A')  
plt.plot(df5['X'][0:11]-273,df5['eta_ex_total'][0:11],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(10,50)
plt.xlim(54, 66)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$T_{cond}$ [$\degree$C]')
plt.ylabel(r'$\eta_{ex}$ [%]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig5b')
plt.show()
plt.close()
  
#########################
##### Figure 5c #######
#########################   
fig, ax = plt.subplots(figsize=(7.5,4)) 
plt.plot(df1['X'][0:11]-273,df1['UCP_H'][0:11],'ro-',label=r'R-32')    
plt.plot(df2['X'][0:11]-273,df2['UCP_H'][0:11],'bs-',label=r'R-290')  
plt.plot(df3['X'][0:11]-273,df3['UCP_H'][0:11],'g^-',label=r'R-410A')  
plt.plot(df4['X'][0:11]-273,df4['UCP_H'][0:11],'yd-',label=r'R-454A')  
plt.plot(df5['X'][0:11]-273,df5['UCP_H'][0:11],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(0.2,0.5)
plt.xlim(54, 66)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$T_{cond}$ [$\degree$C]')
plt.ylabel(r'UCH [$\$$ kWh$^{-1}$]') #{\textdegree}C
leg = plt.legend(fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig5c')
plt.show()
plt.close()
  
  
  
#########################
##### Figure 6 #######
#########################
#import data from excel file
df1 = pd.read_excel('Base_R32.xlsx',sheet_name='deltaT_sup_evap',header=0) #file name
df2 = pd.read_excel('Base_R290.xlsx',sheet_name='deltaT_sup_evap',header=0) #file name
df3 = pd.read_excel('Base_R410A.xlsx',sheet_name='deltaT_sup_evap',header=0) #file name
df4 = pd.read_excel('Base_R454A.xlsx',sheet_name='deltaT_sup_evap',header=0) #file name
df5 = pd.read_excel('Base_R452B.xlsx',sheet_name='deltaT_sup_evap',header=0) #file name  
      
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
      
      
p1, = host.plot(df1['X'],df1['COP_heating'],'bo-',label=r'R-32')
host.plot(df2['X'],df2['COP_heating'],'bs-',label=r'R-290')  
host.plot(df3['X'],df3['COP_heating'],'b^-',label=r'R-410A')  
host.plot(df4['X'],df4['COP_heating'],'bd-',label=r'R-454A')  
host.plot(df5['X'],df5['COP_heating'],'P-',color='b',label=r'R-452B')
      
p2, = par1.plot(df1['X'],df1['eta_ex_total'],'go-',label=r'R-32')
par1.plot(df2['X'],df2['eta_ex_total'],'gs-',label=r'R-290')  
par1.plot(df3['X'],df3['eta_ex_total'],'g^-',label=r'R-410A')  
par1.plot(df4['X'],df4['eta_ex_total'],'gd-',label=r'R-454A')  
par1.plot(df5['X'],df5['eta_ex_total'],'P-',color='g',label=r'R-452B')   
      
p3, = par2.plot(df1['X'],df1['UCP_H'],'ro-',label=r'R-32')
par2.plot(df2['X'],df2['UCP_H'],'rs-',label=r'R-290')  
par2.plot(df3['X'],df3['UCP_H'],'r^-',label=r'R-410A')  
par2.plot(df4['X'],df4['UCP_H'],'rd-',label=r'R-454A')  
par2.plot(df5['X'],df5['UCP_H'],'P-',color='r',label=r'R-452B')  
      
      
host.set_xlim(2,20)
host.set_ylim(2.0, 3.2)
par1.set_ylim(10, 90)
par2.set_ylim(0, 0.6)
      
host.set_xlabel(r'$Sh_{evap}$ [$\degree$C]')
host.set_ylabel(r'$\mathrm{COP_{H}}$ [$-$]')
par1.set_ylabel(r'$\eta_{ex}$ [%]')
par2.set_ylabel(r'UCH [$\$$ kWh$^{-1}$]')
      
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
      
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='k', label='R-32'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='k', label='R-290'),
                   Line2D([0], [0], marker='^', color='w', markerfacecolor='k', label='R-410A'),
                   Line2D([0], [0], marker='d', color='w', markerfacecolor='k', label='R-454A'),
                   Line2D([0], [0], marker='P', color='w', markerfacecolor='k', label='R-452B'),
                   ]
      
      
leg = host.legend(handles=legend_elements,fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig6')
plt.show()
plt.close()
    
#########################
##### Figure 6a #######
#########################
plt.plot(df1['X'],df1['COP_heating'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['COP_heating'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['COP_heating'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['COP_heating'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['COP_heating'],'P-',color='k',label=r'R-452B')   
    
      
# plt.ylim(100,600)
plt.xlim(2,20)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$Sh_{evap}$ [$\degree$C]')
plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig6a')
plt.show()
plt.close()
    
#########################
##### Figure 6b #######
#########################
plt.plot(df1['X'],df1['eta_ex_total'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['eta_ex_total'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['eta_ex_total'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['eta_ex_total'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['eta_ex_total'],'P-',color='k',label=r'R-452B')   
    
      
# plt.ylim(100,600)
plt.xlim(2,20)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$Sh_{evap}$ [$\degree$C]')
plt.ylabel(r'$\eta_{ex}$ [%]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig6b')
plt.show()
plt.close()
    
#########################
##### Figure 6c #######
#########################    
fig, ax = plt.subplots(figsize=(7.5,4)) 
plt.plot(df1['X'],df1['UCP_H'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['UCP_H'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['UCP_H'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['UCP_H'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['UCP_H'],'P-',color='k',label=r'R-452B')   
    
      
# plt.ylim(100,600)
plt.xlim(2,20)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$Sh_{evap}$ [$\degree$C]')
plt.ylabel(r'UCH [$\$$ kWh$^{-1}$]') #{\textdegree}C
leg = plt.legend(fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig6c')
plt.show()
plt.close()
  
  
#########################
##### Figure 7 #######
#########################
#import data from excel file
df1 = pd.read_excel('Base_R32.xlsx',sheet_name='deltaT_sub_cond',header=0) #file name
df2 = pd.read_excel('Base_R290.xlsx',sheet_name='deltaT_sub_cond',header=0) #file name
df3 = pd.read_excel('Base_R410A.xlsx',sheet_name='deltaT_sub_cond',header=0) #file name
df4 = pd.read_excel('Base_R454A.xlsx',sheet_name='deltaT_sub_cond',header=0) #file name
df5 = pd.read_excel('Base_R452B.xlsx',sheet_name='deltaT_sub_cond',header=0) #file name  
   
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
   
   
p1, = host.plot(df1['X'],df1['COP_heating'],'bo-',label=r'R-32')
host.plot(df2['X'],df2['COP_heating'],'bs-',label=r'R-290')  
host.plot(df3['X'],df3['COP_heating'],'b^-',label=r'R-410A')  
host.plot(df4['X'],df4['COP_heating'],'bd-',label=r'R-454A')  
host.plot(df5['X'],df5['COP_heating'],'P-',color='b',label=r'R-452B')
   
p2, = par1.plot(df1['X'],df1['eta_ex_total'],'go-',label=r'R-32')
par1.plot(df2['X'],df2['eta_ex_total'],'gs-',label=r'R-290')  
par1.plot(df3['X'],df3['eta_ex_total'],'g^-',label=r'R-410A')  
par1.plot(df4['X'],df4['eta_ex_total'],'gd-',label=r'R-454A')  
par1.plot(df5['X'],df5['eta_ex_total'],'P-',color='g',label=r'R-452B')   
   
p3, = par2.plot(df1['X'],df1['UCP_H'],'ro-',label=r'R-32')
par2.plot(df2['X'],df2['UCP_H'],'rs-',label=r'R-290')  
par2.plot(df3['X'],df3['UCP_H'],'r^-',label=r'R-410A')  
par2.plot(df4['X'],df4['UCP_H'],'rd-',label=r'R-454A')  
par2.plot(df5['X'],df5['UCP_H'],'P-',color='r',label=r'R-452B')  
   
   
host.set_xlim(2,20)
host.set_ylim(2.6, 3.3)
par1.set_ylim(10, 90)
par2.set_ylim(0, 0.5)
   
host.set_xlabel(r'$Sc_{cond}$ [$\degree$C]')
host.set_ylabel(r'$\mathrm{COP_{H}}$ [$-$]')
par1.set_ylabel(r'$\eta_{ex}$ [%]')
par2.set_ylabel(r'UCH [$\$$ kWh$^{-1}$]')
   
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
   
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='k', label='R-32'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='k', label='R-290'),
                   Line2D([0], [0], marker='^', color='w', markerfacecolor='k', label='R-410A'),
                   Line2D([0], [0], marker='d', color='w', markerfacecolor='k', label='R-454A'),
                   Line2D([0], [0], marker='P', color='w', markerfacecolor='k', label='R-452B'),
                   ]
   
   
leg = host.legend(handles=legend_elements,fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig7')
plt.show()
plt.close()
  
#########################
##### Figure 7a #######
#########################
plt.plot(df1['X'],df1['COP_heating'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['COP_heating'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['COP_heating'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['COP_heating'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['COP_heating'],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(100,600)
plt.xlim(2,20)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$Sc_{cond}$ [$\degree$C]')
plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig7a')
plt.show()
plt.close()
  
#########################
##### Figure 7b #######
#########################
plt.plot(df1['X'],df1['eta_ex_total'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['eta_ex_total'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['eta_ex_total'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['eta_ex_total'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['eta_ex_total'],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(100,600)
plt.xlim(2,20)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$Sc_{cond}$ [$\degree$C]')
plt.ylabel(r'$\eta_{ex}$ [%]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig7b')
plt.show()
plt.close()
  
#########################
##### Figure 7c #######
#########################  
fig, ax = plt.subplots(figsize=(7.5,4))   
plt.plot(df1['X'],df1['UCP_H'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['UCP_H'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['UCP_H'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['UCP_H'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['UCP_H'],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(100,600)
plt.xlim(2,20)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$Sc_{cond}$ [$\degree$C]')
plt.ylabel(r'UCH [$\$$ kWh$^{-1}$]') #{\textdegree}C
leg = plt.legend(fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig7c')
plt.show()
plt.close()
 
 
#########################
##### Figure 8 #######
#########################
#import data from excel file
df1 = pd.read_excel('Base_R32.xlsx',sheet_name='TTD',header=0) #file name
df2 = pd.read_excel('Base_R290.xlsx',sheet_name='TTD',header=0) #file name
df3 = pd.read_excel('Base_R410A.xlsx',sheet_name='TTD',header=0) #file name
df4 = pd.read_excel('Base_R454A.xlsx',sheet_name='TTD',header=0) #file name
df5 = pd.read_excel('Base_R452B.xlsx',sheet_name='TTD',header=0) #file name  
  
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
  
  
p1, = host.plot(df1['X'],df1['COP_heating'],'bo-',label=r'R-32')
host.plot(df2['X'],df2['COP_heating'],'bs-',label=r'R-290')  
host.plot(df3['X'],df3['COP_heating'],'b^-',label=r'R-410A')  
host.plot(df4['X'],df4['COP_heating'],'bd-',label=r'R-454A')  
host.plot(df5['X'],df5['COP_heating'],'P-',color='b',label=r'R-452B')
  
p2, = par1.plot(df1['X'],df1['eta_ex_total'],'go-',label=r'R-32')
par1.plot(df2['X'],df2['eta_ex_total'],'gs-',label=r'R-290')  
par1.plot(df3['X'],df3['eta_ex_total'],'g^-',label=r'R-410A')  
par1.plot(df4['X'],df4['eta_ex_total'],'gd-',label=r'R-454A')  
par1.plot(df5['X'],df5['eta_ex_total'],'P-',color='g',label=r'R-452B')   
  
p3, = par2.plot(df1['X'],df1['UCP_H'],'ro-',label=r'R-32')
par2.plot(df2['X'],df2['UCP_H'],'rs-',label=r'R-290')  
par2.plot(df3['X'],df3['UCP_H'],'r^-',label=r'R-410A')  
par2.plot(df4['X'],df4['UCP_H'],'rd-',label=r'R-454A')  
par2.plot(df5['X'],df5['UCP_H'],'P-',color='r',label=r'R-452B')  
  
  
host.set_xlim(2,20)
host.set_ylim(2.0, 3.2)
par1.set_ylim(10, 90)
par2.set_ylim(0, 0.7)
  
host.set_xlabel(r'$TTD$ [$\degree$C]')
host.set_ylabel(r'$\mathrm{COP_{H}}$ [$-$]')
par1.set_ylabel(r'$\eta_{ex}$ [%]')
par2.set_ylabel(r'UCH [$\$$ kWh$^{-1}$]')
  
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
  
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='k', label='R-32'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='k', label='R-290'),
                   Line2D([0], [0], marker='^', color='w', markerfacecolor='k', label='R-410A'),
                   Line2D([0], [0], marker='d', color='w', markerfacecolor='k', label='R-454A'),
                   Line2D([0], [0], marker='P', color='w', markerfacecolor='k', label='R-452B'),
                   ]
  
  
leg = host.legend(handles=legend_elements,fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig8')
plt.show()
plt.close()
  
#########################
##### Figure 8a #######
#########################
plt.plot(df1['X'],df1['COP_heating'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['COP_heating'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['COP_heating'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['COP_heating'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['COP_heating'],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(100,600)
plt.xlim(2,20)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$TTD$ [$\degree$C]')
plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig8a')
plt.show()
plt.close()
  
#########################
##### Figure 8b #######
#########################
plt.plot(df1['X'],df1['eta_ex_total'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['eta_ex_total'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['eta_ex_total'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['eta_ex_total'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['eta_ex_total'],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(100,600)
plt.xlim(2,20)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$TTD$ [$\degree$C]')
plt.ylabel(r'$\eta_{ex}$ [%]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig8b')
plt.show()
plt.close()
  
#########################
##### Figure 8c #######
######################### 
fig, ax = plt.subplots(figsize=(7.5,4))    
plt.plot(df1['X'],df1['UCP_H'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['UCP_H'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['UCP_H'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['UCP_H'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['UCP_H'],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(100,600)
plt.xlim(2,20)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$TTD$ [$\degree$C]')
plt.ylabel(r'UCH [$\$$ kWh$^{-1}$]') #{\textdegree}C
leg = plt.legend(fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig8c')
plt.show()
plt.close()
 
 
#########################
##### Figure 9 #######
#########################
#import data from excel file
df1 = pd.read_excel('Base_R32.xlsx',sheet_name='c_eva1',header=0) #file name
df2 = pd.read_excel('Base_R290.xlsx',sheet_name='c_eva1',header=0) #file name
df3 = pd.read_excel('Base_R410A.xlsx',sheet_name='c_eva1',header=0) #file name
df4 = pd.read_excel('Base_R454A.xlsx',sheet_name='c_eva1',header=0) #file name
df5 = pd.read_excel('Base_R452B.xlsx',sheet_name='c_eva1',header=0) #file name  
  
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
  
  
p1, = host.plot(df1['X'],df1['COP_heating'],'bo-',label=r'R-32')
host.plot(df2['X'],df2['COP_heating'],'bs-',label=r'R-290')  
host.plot(df3['X'],df3['COP_heating'],'b^-',label=r'R-410A')  
host.plot(df4['X'],df4['COP_heating'],'bd-',label=r'R-454A')  
host.plot(df5['X'],df5['COP_heating'],'P-',color='b',label=r'R-452B')
  
p2, = par1.plot(df1['X'],df1['eta_ex_total'],'go-',label=r'R-32')
par1.plot(df2['X'],df2['eta_ex_total'],'gs-',label=r'R-290')  
par1.plot(df3['X'],df3['eta_ex_total'],'g^-',label=r'R-410A')  
par1.plot(df4['X'],df4['eta_ex_total'],'gd-',label=r'R-454A')  
par1.plot(df5['X'],df5['eta_ex_total'],'P-',color='g',label=r'R-452B')   
  
p3, = par2.plot(df1['X'],df1['UCP_H'],'ro-',label=r'R-32')
par2.plot(df2['X'],df2['UCP_H'],'rs-',label=r'R-290')  
par2.plot(df3['X'],df3['UCP_H'],'r^-',label=r'R-410A')  
par2.plot(df4['X'],df4['UCP_H'],'rd-',label=r'R-454A')  
par2.plot(df5['X'],df5['UCP_H'],'P-',color='r',label=r'R-452B')  
  
  
host.set_xlim(0,1)
host.set_ylim(2.0, 3.2)
par1.set_ylim(10, 90)
par2.set_ylim(0, 0.7)
  
host.set_xlabel(r'$GCCLS$ [$-$]')
host.set_ylabel(r'$\mathrm{COP_{H}}$ [$-$]')
par1.set_ylabel(r'$\eta_{ex}$ [%]')
par2.set_ylabel(r'UCH [$\$$ kWh$^{-1}$]')
  
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
  
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='k', label='R-32'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='k', label='R-290'),
                   Line2D([0], [0], marker='^', color='w', markerfacecolor='k', label='R-410A'),
                   Line2D([0], [0], marker='d', color='w', markerfacecolor='k', label='R-454A'),
                   Line2D([0], [0], marker='P', color='w', markerfacecolor='k', label='R-452B'),
                   ]
  
  
leg = host.legend(handles=legend_elements,fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig9')
plt.show()
plt.close()
  
#########################
##### Figure 9a #######
#########################
plt.plot(df1['X'],df1['COP_heating'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['COP_heating'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['COP_heating'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['COP_heating'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['COP_heating'],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(100,600)
plt.xlim(0,1)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$GCCLS$ [$-$]')
plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig9a')
plt.show()
plt.close()
  
#########################
##### Figure 9b #######
#########################
plt.plot(df1['X'],df1['eta_ex_total'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['eta_ex_total'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['eta_ex_total'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['eta_ex_total'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['eta_ex_total'],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(100,600)
plt.xlim(0,1)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$GCCLS$ [$-$]')
plt.ylabel(r'$\eta_{ex}$ [%]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig9b')
plt.show()
plt.close()
  
#########################
##### Figure 9c #######
#########################  
fig, ax = plt.subplots(figsize=(7.5,4))   
plt.plot(df1['X'],df1['UCP_H'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['UCP_H'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['UCP_H'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['UCP_H'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['UCP_H'],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(100,600)
plt.xlim(0,1)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$GCCLS$ [$-$]')
plt.ylabel(r'UCH [$\$$ kWh$^{-1}$]') #{\textdegree}C
leg = plt.legend(fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig9c')
plt.show()
plt.close()
 
 
#########################
##### Figure 10 #######
#########################
#import data from excel file
df1 = pd.read_excel('Base_R32.xlsx',sheet_name='c_cond1',header=0) #file name
df2 = pd.read_excel('Base_R290.xlsx',sheet_name='c_cond1',header=0) #file name
df3 = pd.read_excel('Base_R410A.xlsx',sheet_name='c_cond1',header=0) #file name
df4 = pd.read_excel('Base_R454A.xlsx',sheet_name='c_cond1',header=0) #file name
df5 = pd.read_excel('Base_R452B.xlsx',sheet_name='c_cond1',header=0) #file name  
  
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
  
  
p1, = host.plot(df1['X'],df1['COP_heating'],'bo-',label=r'R-32')
host.plot(df2['X'],df2['COP_heating'],'bs-',label=r'R-290')  
host.plot(df3['X'],df3['COP_heating'],'b^-',label=r'R-410A')  
host.plot(df4['X'],df4['COP_heating'],'bd-',label=r'R-454A')  
host.plot(df5['X'],df5['COP_heating'],'P-',color='b',label=r'R-452B')
  
p2, = par1.plot(df1['X'],df1['eta_ex_total'],'go-',label=r'R-32')
par1.plot(df2['X'],df2['eta_ex_total'],'gs-',label=r'R-290')  
par1.plot(df3['X'],df3['eta_ex_total'],'g^-',label=r'R-410A')  
par1.plot(df4['X'],df4['eta_ex_total'],'gd-',label=r'R-454A')  
par1.plot(df5['X'],df5['eta_ex_total'],'P-',color='g',label=r'R-452B')   
  
p3, = par2.plot(df1['X'],df1['UCP_H'],'ro-',label=r'R-32')
par2.plot(df2['X'],df2['UCP_H'],'rs-',label=r'R-290')  
par2.plot(df3['X'],df3['UCP_H'],'r^-',label=r'R-410A')  
par2.plot(df4['X'],df4['UCP_H'],'rd-',label=r'R-454A')  
par2.plot(df5['X'],df5['UCP_H'],'P-',color='r',label=r'R-452B')  
  
  
host.set_xlim(0,1)
host.set_ylim(2.0, 3.2)
par1.set_ylim(10, 90)
par2.set_ylim(0, 0.7)
  
host.set_xlabel(r'$GCHLS$ [$-$]')
host.set_ylabel(r'$\mathrm{COP_{H}}$ [$-$]')
par1.set_ylabel(r'$\eta_{ex}$ [%]')
par2.set_ylabel(r'UCH [$\$$ kWh$^{-1}$]')
  
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
  
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='k', label='R-32'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='k', label='R-290'),
                   Line2D([0], [0], marker='^', color='w', markerfacecolor='k', label='R-410A'),
                   Line2D([0], [0], marker='d', color='w', markerfacecolor='k', label='R-454A'),
                   Line2D([0], [0], marker='P', color='w', markerfacecolor='k', label='R-452B'),
                   ]
  
  
leg = host.legend(handles=legend_elements,fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig10')
plt.show()
plt.close()
  
#########################
##### Figure 10a #######
#########################
plt.plot(df1['X'],df1['COP_heating'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['COP_heating'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['COP_heating'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['COP_heating'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['COP_heating'],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(100,600)
plt.xlim(0,1)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$GCHLS$ [$-$]')
plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig10a')
plt.show()
plt.close()
  
#########################
##### Figure 10b #######
#########################
plt.plot(df1['X'],df1['eta_ex_total'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['eta_ex_total'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['eta_ex_total'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['eta_ex_total'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['eta_ex_total'],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(100,600)
plt.xlim(0,1)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$GCHLS$ [$-$]')
plt.ylabel(r'$\eta_{ex}$ [%]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig10b')
plt.show()
plt.close()
  
#########################
##### Figure 10c #######
#########################
fig, ax = plt.subplots(figsize=(7.5,4))     
plt.plot(df1['X'],df1['UCP_H'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['UCP_H'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['UCP_H'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['UCP_H'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['UCP_H'],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(100,600)
plt.xlim(0,1)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$GCHLS$ [$-$]')
plt.ylabel(r'UCH [$\$$ kWh$^{-1}$]') #{\textdegree}C
leg = plt.legend(fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig10c')
plt.show()
plt.close()
 
 
 
#########################
##### Figure 11 #######
#########################
#import data from excel file
df1 = pd.read_excel('Base_R32.xlsx',sheet_name='deltaT_sup_evap',header=0) #file name
df2 = pd.read_excel('Base_R290.xlsx',sheet_name='deltaT_sup_evap',header=0) #file name
df3 = pd.read_excel('Base_R410A.xlsx',sheet_name='deltaT_sup_evap',header=0) #file name
df4 = pd.read_excel('Base_R454A.xlsx',sheet_name='deltaT_sup_evap',header=0) #file name
df5 = pd.read_excel('Base_R452B.xlsx',sheet_name='deltaT_sup_evap',header=0) #file name  
       
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
       
       
p1, = host.plot(df1['X'],df1['COP_heating'],'bo-',label=r'R-32')
host.plot(df2['X'],df2['COP_heating'],'bs-',label=r'R-290')  
host.plot(df3['X'],df3['COP_heating'],'b^-',label=r'R-410A')  
host.plot(df4['X'],df4['COP_heating'],'bd-',label=r'R-454A')  
host.plot(df5['X'],df5['COP_heating'],'P-',color='b',label=r'R-452B')
       
p2, = par1.plot(df1['X'],df1['eta_ex_total'],'go-',label=r'R-32')
par1.plot(df2['X'],df2['eta_ex_total'],'gs-',label=r'R-290')  
par1.plot(df3['X'],df3['eta_ex_total'],'g^-',label=r'R-410A')  
par1.plot(df4['X'],df4['eta_ex_total'],'gd-',label=r'R-454A')  
par1.plot(df5['X'],df5['eta_ex_total'],'P-',color='g',label=r'R-452B')   
       
p3, = par2.plot(df1['X'],df1['UCP_H'],'ro-',label=r'R-32')
par2.plot(df2['X'],df2['UCP_H'],'rs-',label=r'R-290')  
par2.plot(df3['X'],df3['UCP_H'],'r^-',label=r'R-410A')  
par2.plot(df4['X'],df4['UCP_H'],'rd-',label=r'R-454A')  
par2.plot(df5['X'],df5['UCP_H'],'P-',color='r',label=r'R-452B')  
       
       
host.set_xlim(2,20)
host.set_ylim(2.0, 3.2)
par1.set_ylim(10, 90)
par2.set_ylim(0, 0.6)
       
host.set_xlabel(r'$Sh_{econ}$ [$\degree$C]')
host.set_ylabel(r'$\mathrm{COP_{H}}$ [$-$]')
par1.set_ylabel(r'$\eta_{ex}$ [%]')
par2.set_ylabel(r'UCH [$\$$ kWh$^{-1}$]')
       
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
       
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='k', label='R-32'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='k', label='R-290'),
                   Line2D([0], [0], marker='^', color='w', markerfacecolor='k', label='R-410A'),
                   Line2D([0], [0], marker='d', color='w', markerfacecolor='k', label='R-454A'),
                   Line2D([0], [0], marker='P', color='w', markerfacecolor='k', label='R-452B'),
                   ]
       
       
leg = host.legend(handles=legend_elements,fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig11')
plt.show()
plt.close()
    
#########################
##### Figure 11a #######
#########################
plt.plot(df1['X'],df1['COP_heating'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['COP_heating'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['COP_heating'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['COP_heating'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['COP_heating'],'P-',color='k',label=r'R-452B')   
    
      
# plt.ylim(100,600)
plt.xlim(2,20)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$Sh_{econ}$ [$\degree$C]')
plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig11a')
plt.show()
plt.close()
    
#########################
##### Figure 11b #######
#########################
plt.plot(df1['X'],df1['eta_ex_total'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['eta_ex_total'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['eta_ex_total'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['eta_ex_total'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['eta_ex_total'],'P-',color='k',label=r'R-452B')   
    
      
# plt.ylim(100,600)
plt.xlim(2,20)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$Sh_{econ}$ [$\degree$C]')
plt.ylabel(r'$\eta_{ex}$ [%]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig11b')
plt.show()
plt.close()
    
#########################
##### Figure 11c #######
#########################    
fig, ax = plt.subplots(figsize=(7.5,4)) 
plt.plot(df1['X'],df1['UCP_H'],'ro-',label=r'R-32')    
plt.plot(df2['X'],df2['UCP_H'],'bs-',label=r'R-290')  
plt.plot(df3['X'],df3['UCP_H'],'g^-',label=r'R-410A')  
plt.plot(df4['X'],df4['UCP_H'],'yd-',label=r'R-454A')  
plt.plot(df5['X'],df5['UCP_H'],'P-',color='k',label=r'R-452B')   
    
      
# plt.ylim(100,600)
plt.xlim(2,20)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$Sh_{econ}$ [$\degree$C]')
plt.ylabel(r'UCH [$\$$ kWh$^{-1}$]') #{\textdegree}C
leg = plt.legend(fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig11c')
plt.show()
plt.close()




#########################
##### Figure 13 plot #######
#########################  
#import data from excel file
df = pd.read_excel('bar.xlsx',sheet_name='Sheet1',header=0) #file name 
       
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
y3 = df['UCH']
 
host.bar(np.arange(1,6,1)-0.2,y1,width=0.2,color='b',linewidth=0.9,align='center',alpha=0.9,label=r'$\mathrm{COP_{H}}$',hatch=5*'\\')    
    
par1.bar(np.arange(1,6,1),y2,width=0.2,color='g',linewidth=0.9,align='center',alpha=0.9,label=r'$\eta_{ex}$',hatch=2*'x')  
        
par2.bar(np.arange(1,6,1)+0.2,y3,width=0.2,color='r',linewidth=0.9,align='center',alpha=0.9,label=r'UCH',hatch=2*'//') 
 
# host.set_xlim(2,20)
host.set_ylim(0, 5)
# par1.set_ylim(10, 90)
# par2.set_ylim(0, 0.6)
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
host.set_ylabel(r'$\mathrm{COP_{H}}$ [$-$]')
par1.set_ylabel(r'$\eta_{ex}$ [%]')
par2.set_ylabel(r'UCH [$\$$ kWh$^{-1}$]')
       
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
       
legend_elements = [Patch(facecolor='blue', edgecolor='k',hatch=5*'\\',label='$\mathrm{COP_{H}}$'),
                   Patch(facecolor='green', edgecolor='k',hatch=3*'x',label='$\eta_{ex}$'),
                   Patch(facecolor='red', edgecolor='k',hatch=2*'//',label='UCH'),
                   ]
 
       
leg = host.legend(handles=legend_elements,fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig13_plot')
plt.show()
plt.close()



#########################
##### Figure 13table  #######
######################### 
cell_text = [[0.46,0.61,0.66,0.42,0.31],
[0.32,0.49,0.69,0.53,0.47],
[2.83,2.48,2.07,2.63,1.33],
[55.04,55.68,55.10,55.37,55.52],
[8.17,9.37,8.91,9.48,7.47],
[9.82,9.87,9.86,9.83,9.89],
[7.76,7.96,7.94,7.93,7.54],
[35.88,35.480,36.14,34.70,36.15],
[26.90,26.90,26.90,26.90,26.90],
[26.90,26.90,26.90,26.90,26.90],
[35.88,35.48,36.14,34.70,36.15],
[1.80,2.32,2.51,1.63,2.29],
[0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00],
[3.92,3.93,4.12,3.60,4.10],
[4.97,4.56,5.03,4.11,5.06],
[3.72,1.76,2.69,2.04,3.00]]

rows = [r'$GCCLS$ [$-$]',
r'$GCHLS$ [$-$]',
r'$T_{evap}$ [$\degree$C]',
r'$T_{cond}$ [$\degree$C]',
r'$DT_{econ}$ [$\degree$C]',
r'$Sc_{cond}$ [$\degree$C]',
r'$Sh_{evap}$ [$\degree$C]',
r'$\dot Q_{InHX}$ [kW]',
r'$\dot Q_{OutHX}$ [kW]',
r'$\dot Q_{evap}$  [kW]',
r'$\dot Q_{cond}$ [kW]',
r'$\dot Q_{econ}$ [kW]',
r'$\dot W_{pump,1}$ [kW]',
r'$\dot W_{pump,2}$ [kW]',
r'$\dot W_{comp,1}$ [kW]',
r'$\dot W_{comp,2}$ [kW]',
r'$\dot{Ex}_{P}^{InHX}$ [kW]']



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
    
savefigs('fig13_table')
plt.show()
plt.close()
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



# #########################
# ##### Figure 5 #######
# #########################
# #import data from excel file
df1 = pd.read_excel('Base_R32.xlsx',sheet_name='T_cond',header=0) #file name
df2 = pd.read_excel('Base_R290.xlsx',sheet_name='T_cond',header=0) #file name
df3 = pd.read_excel('Base_R410A.xlsx',sheet_name='T_cond',header=0) #file name
df4 = pd.read_excel('Base_R454A.xlsx',sheet_name='T_cond',header=0) #file name
df5 = pd.read_excel('Base_R452B.xlsx',sheet_name='T_cond',header=0) #file name  
#   
# def make_patch_spines_invisible(ax):
#     ax.set_frame_on(True)
#     ax.patch.set_visible(False)
#     for sp in ax.spines.values():
#         sp.set_visible(False)
#   
# fig, host = plt.subplots(figsize=(9,4))
# fig.subplots_adjust(right=0.7)
#   
# par1 = host.twinx()
# par2 = host.twinx()
#   
# par1.spines["left"].set_position(("axes", -0.18)) # green one
# par2.spines["left"].set_position(("axes", -0.35)) # red one
# make_patch_spines_invisible(par1)
# make_patch_spines_invisible(par2)
# par1.spines["left"].set_visible(True)
# par1.yaxis.set_label_position('left')
# par1.yaxis.set_ticks_position('left')
# par2.spines["left"].set_visible(True)
# par2.yaxis.set_label_position('left')
# par2.yaxis.set_ticks_position('left')
#   
#   
# p1, = host.plot(df1['X'][0:11]-273,df1['COP_heating'][0:11],'bo-',label=r'R-32')
# host.plot(df2['X'][0:11]-273,df2['COP_heating'][0:11],'bs-',label=r'R-290')  
# host.plot(df3['X'][0:11]-273,df3['COP_heating'][0:11],'b^-',label=r'R-410A')  
# host.plot(df4['X'][0:11]-273,df4['COP_heating'][0:11],'bd-',label=r'R-454A')  
# host.plot(df5['X'][0:11]-273,df5['COP_heating'][0:11],'P-',color='b',label=r'R-452B')
#   
# p2, = par1.plot(df1['X'][0:11]-273,df1['eta_ex_total'][0:11],'go-',label=r'R-32')
# par1.plot(df2['X'][0:11]-273,df2['eta_ex_total'][0:11],'gs-',label=r'R-290')  
# par1.plot(df3['X'][0:11]-273,df3['eta_ex_total'][0:11],'g^-',label=r'R-410A')  
# par1.plot(df4['X'][0:11]-273,df4['eta_ex_total'][0:11],'gd-',label=r'R-454A')  
# par1.plot(df5['X'][0:11]-273,df5['eta_ex_total'][0:11],'P-',color='g',label=r'R-452B')   
#   
# p3, = par2.plot(df1['X'][0:11]-273,df1['UCP_H'][0:11],'ro-',label=r'R-32')
# par2.plot(df2['X'][0:11]-273,df2['UCP_H'][0:11],'rs-',label=r'R-290')  
# par2.plot(df3['X'][0:11]-273,df3['UCP_H'][0:11],'r^-',label=r'R-410A')  
# par2.plot(df4['X'][0:11]-273,df4['UCP_H'][0:11],'rd-',label=r'R-454A')  
# par2.plot(df5['X'][0:11]-273,df5['UCP_H'][0:11],'P-',color='r',label=r'R-452B')  
#   
#   
# host.set_xlim(54, 66)
# host.set_ylim(2.6, 4)
# par1.set_ylim(10, 90)
# par2.set_ylim(0, 0.5)
#   
# host.set_xlabel(r'$T_{cond}$ [$\degree$C]')
# host.set_ylabel(r'$\mathrm{COP_{H}}$ [$-$]')
# par1.set_ylabel(r'$\eta_{ex}$ [%]')
# par2.set_ylabel(r'UCH [$\$$ kWh$^{-1}$]')
#   
# host.yaxis.label.set_color('blue')
# par1.yaxis.label.set_color('green')
# par2.yaxis.label.set_color('red')
#   
# tkw = dict(size=4, width=1.5)
# host.tick_params(axis='y', colors='blue', **tkw)
# par1.tick_params(axis='y', colors='green', **tkw)
# par2.tick_params(axis='y', colors='red', **tkw)
# host.tick_params(axis='x', **tkw)
#   
# from matplotlib.patches import Patch
# from matplotlib.lines import Line2D
#   
# legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='k', label='R-32'),
#                    Line2D([0], [0], marker='s', color='w', markerfacecolor='k', label='R-290'),
#                    Line2D([0], [0], marker='^', color='w', markerfacecolor='k', label='R-410A'),
#                    Line2D([0], [0], marker='d', color='w', markerfacecolor='k', label='R-454A'),
#                    Line2D([0], [0], marker='P', color='w', markerfacecolor='k', label='R-452B'),
#                    ]
#   
#   
# leg = host.legend(handles=legend_elements,fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig5')
# plt.show()
# plt.close()
#   
# #########################
# ##### Figure 5a #######
# #########################
# plt.plot(df1['X'][0:11]-273,df1['COP_heating'][0:11],'ro-',label=r'R-32')    
# plt.plot(df2['X'][0:11]-273,df2['COP_heating'][0:11],'bs-',label=r'R-290')  
# plt.plot(df3['X'][0:11]-273,df3['COP_heating'][0:11],'g^-',label=r'R-410A')  
# plt.plot(df4['X'][0:11]-273,df4['COP_heating'][0:11],'yd-',label=r'R-454A')  
# plt.plot(df5['X'][0:11]-273,df5['COP_heating'][0:11],'P-',color='k',label=r'R-452B')   
#   
#     
# plt.ylim(2.6,3.8)
# plt.xlim(54, 66)
# # plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# #            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# # plt.tick_params(
# #     axis='x',          # changes apply to the x-axis
# #     which='both',      # both major and minor ticks are affected
# #     bottom='on',      # ticks along the bottom edge are on
# #     top='off',         # ticks along the top edge are off
# #     labelbottom='on') # labels along the bottom edge are off
# plt.xlabel(r'$T_{cond}$ [$\degree$C]')
# plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
# # leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# # frame  = leg.get_frame()  
# # frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig5a')
# plt.show()
# plt.close()
#   
# #########################
# ##### Figure 5b #######
# #########################
# plt.plot(df1['X'][0:11]-273,df1['eta_ex_total'][0:11],'ro-',label=r'R-32')    
# plt.plot(df2['X'][0:11]-273,df2['eta_ex_total'][0:11],'bs-',label=r'R-290')  
# plt.plot(df3['X'][0:11]-273,df3['eta_ex_total'][0:11],'g^-',label=r'R-410A')  
# plt.plot(df4['X'][0:11]-273,df4['eta_ex_total'][0:11],'yd-',label=r'R-454A')  
# plt.plot(df5['X'][0:11]-273,df5['eta_ex_total'][0:11],'P-',color='k',label=r'R-452B')   
#   
#     
# # plt.ylim(10,50)
# plt.xlim(54, 66)
# # plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# #            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# # plt.tick_params(
# #     axis='x',          # changes apply to the x-axis
# #     which='both',      # both major and minor ticks are affected
# #     bottom='on',      # ticks along the bottom edge are on
# #     top='off',         # ticks along the top edge are off
# #     labelbottom='on') # labels along the bottom edge are off
# plt.xlabel(r'$T_{cond}$ [$\degree$C]')
# plt.ylabel(r'$\eta_{ex}$ [%]') #{\textdegree}C
# # leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# # frame  = leg.get_frame()  
# # frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig5b')
# plt.show()
# plt.close()
#   
# #########################
# ##### Figure 5c #######
# #########################   
# fig, ax = plt.subplots(figsize=(7.5,4)) 
# plt.plot(df1['X'][0:11]-273,df1['UCP_H'][0:11],'ro-',label=r'R-32')    
# plt.plot(df2['X'][0:11]-273,df2['UCP_H'][0:11],'bs-',label=r'R-290')  
# plt.plot(df3['X'][0:11]-273,df3['UCP_H'][0:11],'g^-',label=r'R-410A')  
# plt.plot(df4['X'][0:11]-273,df4['UCP_H'][0:11],'yd-',label=r'R-454A')  
# plt.plot(df5['X'][0:11]-273,df5['UCP_H'][0:11],'P-',color='k',label=r'R-452B')   
#   
#     
# # plt.ylim(0.2,0.5)
# plt.xlim(54, 66)
# # plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# #            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# # plt.tick_params(
# #     axis='x',          # changes apply to the x-axis
# #     which='both',      # both major and minor ticks are affected
# #     bottom='on',      # ticks along the bottom edge are on
# #     top='off',         # ticks along the top edge are off
# #     labelbottom='on') # labels along the bottom edge are off
# plt.xlabel(r'$T_{cond}$ [$\degree$C]')
# plt.ylabel(r'UCH [$\$$ kWh$^{-1}$]') #{\textdegree}C
# leg = plt.legend(fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig5c')
# plt.show()
# plt.close()
  
#########################
##### Figure 5new #######
#########################
fig, ax = plt.subplots(figsize=(6,9))
plt.subplot(3, 1, 1)
plt.plot(df1['X'][0:11]-273,df1['COP_heating'][0:11],'ro-',label=r'R-32')    
plt.plot(df2['X'][0:11]-273,df2['COP_heating'][0:11],'bs-',label=r'R-290')  
plt.plot(df3['X'][0:11]-273,df3['COP_heating'][0:11],'g^-',label=r'R-410A')  
plt.plot(df4['X'][0:11]-273,df4['COP_heating'][0:11],'yd-',label=r'R-454A')  
plt.plot(df5['X'][0:11]-273,df5['COP_heating'][0:11],'P-',color='k',label=r'R-452B')   
  
    
plt.ylim(2.6,3.8)
plt.xlim(54, 66)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are on
    top='on',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
# plt.xlabel(r'$T_{cond}$ [$\degree$C]')
plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig5a')
# plt.show()
# plt.close()



plt.subplot(3, 1, 2)
plt.plot(df1['X'][0:11]-273,df1['eta_ex_total'][0:11],'ro-',label=r'R-32')    
plt.plot(df2['X'][0:11]-273,df2['eta_ex_total'][0:11],'bs-',label=r'R-290')  
plt.plot(df3['X'][0:11]-273,df3['eta_ex_total'][0:11],'g^-',label=r'R-410A')  
plt.plot(df4['X'][0:11]-273,df4['eta_ex_total'][0:11],'yd-',label=r'R-454A')  
plt.plot(df5['X'][0:11]-273,df5['eta_ex_total'][0:11],'P-',color='k',label=r'R-452B')   
  
    
# plt.ylim(10,50)
plt.xlim(54, 66)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are on
    top='on',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
# plt.xlabel(r'$T_{cond}$ [$\degree$C]')
plt.ylabel(r'$\eta_{ex}$ [%]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig5b')
# plt.show()
# plt.close()
  


plt.subplot(3, 1, 3)
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
# leg = plt.legend(fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig5new')
plt.show()
plt.close()  
  

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
mpl.rcParams['figure.figsize'] = [6,6]
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


# # #########################
# # ##### Figure 12a #######
# # #########################
# #import data from excel file
# df1 = pd.read_excel('pareto.xlsx',sheet_name='R-32',header=0) #file name
# #row of optimal value in excel file for
# optimal_point = 62
# #draw dashed box
# x=np.array([df1['UCH'].iloc[0],df1['UCH'].iloc[0],df1['UCH'].iloc[-1],df1['UCH'].iloc[-1],df1['UCH'].iloc[0]])
# y=np.array([df1['COP'].iloc[0],df1['COP'].iloc[-1],df1['COP'].iloc[-1],df1['COP'].iloc[0],df1['COP'].iloc[0]])
# plt.plot(x,y,'--k')
# #plot data point
# plt.plot(df1['UCH'],df1['COP'],'ro',label=r'R-32')
# #plot optimal TOPSIS point
# plt.plot(df1['UCH'][optimal_point],df1['COP'][optimal_point],'ko',label=r'R-32')
# #annotation
# plt.annotate('A',xy=(0,0),xytext=(df1['UCH'].iloc[0]-0.0015,df1['COP'].iloc[0]-0.0075),annotation_clip=False)
# plt.annotate('B',xy=(0,0),xytext=(df1['UCH'].iloc[-1]+0.0005,df1['COP'].iloc[-1]-0.0075),annotation_clip=False)
# plt.annotate('AB',xy=(0,0),xytext=(df1['UCH'].iloc[optimal_point]-0.00075,df1['COP'].iloc[optimal_point]+0.01),annotation_clip=False)
# 
# plt.annotate('Ideal Pareto solution',xy=(0,0),xytext=(df1['UCH'].iloc[0]+0.0005,df1['COP'].iloc[-1]-0.03),annotation_clip=False)
# plt.annotate('Non-ideal Pareto solution',xy=(0,0),xytext=(df1['UCH'].iloc[-1]-0.0155,df1['COP'].iloc[0]+0.01),annotation_clip=False)
# 
# plt.annotate('Ideal point for\nminimum UCH', xy=(df1['UCH'].iloc[0],df1['COP'].iloc[0]), xytext=(df1['UCH'].iloc[0]+0.005,df1['COP'].iloc[0]+0.1),
#             arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0)
#             )
# plt.annotate('Ideal point for\nmaximum $\mathrm{COP_H}$', xy=(df1['UCH'].iloc[-1],df1['COP'].iloc[-1]), xytext=(df1['UCH'].iloc[-1]-0.01,df1['COP'].iloc[-1]-0.22),
#             arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0)
#             )
# plt.annotate('Optimal point using\nTOPSIS method', xy=(df1['UCH'].iloc[optimal_point],df1['COP'].iloc[optimal_point]), xytext=(df1['UCH'].iloc[optimal_point]-0.016,df1['COP'].iloc[optimal_point]-0.15),
#             arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0)
#             )
# 
# plt.ylim(3.4,4.08)
# plt.xlim(0.2075,0.235)
# # plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# #            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# # plt.tick_params(
# #     axis='x',          # changes apply to the x-axis
# #     which='both',      # both major and minor ticks are affected
# #     bottom='on',      # ticks along the bottom edge are on
# #     top='off',         # ticks along the top edge are off
# #     labelbottom='on') # labels along the bottom edge are off
# plt.xlabel(r'UCH [\$ kWh$^{-1}$]')
# plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
# # leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# # frame  = leg.get_frame()  
# # frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig12a')
# plt.show()
# plt.close()


# #########################
# ##### Figure 12b #######
# #########################
#import data from excel file
df1 = pd.read_excel('pareto.xlsx',sheet_name='R-290',header=0) #file name
#row of optimal value in excel file for
optimal_point = 62
#draw dashed box
x=np.array([df1['UCH'].iloc[0],df1['UCH'].iloc[0],df1['UCH'].iloc[-1],df1['UCH'].iloc[-1],df1['UCH'].iloc[0]])
y=np.array([df1['COP'].iloc[0],df1['COP'].iloc[-1],df1['COP'].iloc[-1],df1['COP'].iloc[0],df1['COP'].iloc[0]])
plt.plot(x,y,'--k')
#plot data point
plt.plot(df1['UCH'],df1['COP'],'bs',label=r'R-290')
#plot optimal TOPSIS point
plt.plot(df1['UCH'][optimal_point],df1['COP'][optimal_point],'ks',label=r'R-290')
#annotation
plt.annotate('A',xy=(0,0),xytext=(df1['UCH'].iloc[0]-0.002,df1['COP'].iloc[0]-0.0075),annotation_clip=False)
plt.annotate('B',xy=(0,0),xytext=(df1['UCH'].iloc[-1]+0.0005,df1['COP'].iloc[-1]-0.0075),annotation_clip=False)
plt.annotate('AB',xy=(0,0),xytext=(df1['UCH'].iloc[optimal_point]-0.002,df1['COP'].iloc[optimal_point]+0.01),annotation_clip=False)
 
plt.annotate('Ideal Pareto solution',xy=(0,0),xytext=(df1['UCH'].iloc[0]+0.0005,df1['COP'].iloc[-1]-0.03),annotation_clip=False)
plt.annotate('Non-ideal Pareto solution',xy=(0,0),xytext=(df1['UCH'].iloc[-1]-0.021,df1['COP'].iloc[0]+0.01),annotation_clip=False)
 
plt.annotate('Ideal point for\nminimum UCH', xy=(df1['UCH'].iloc[0],df1['COP'].iloc[0]), xytext=(df1['UCH'].iloc[0]+0.01,df1['COP'].iloc[0]+0.1),
            arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0)
            )
plt.annotate('Ideal point for\nmaximum $\mathrm{COP_H}$', xy=(df1['UCH'].iloc[-1],df1['COP'].iloc[-1]), xytext=(df1['UCH'].iloc[-1]-0.0135,df1['COP'].iloc[-1]-0.25),
            arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0)
            )
plt.annotate('Optimal point using\nTOPSIS method', xy=(df1['UCH'].iloc[optimal_point],df1['COP'].iloc[optimal_point]), xytext=(df1['UCH'].iloc[optimal_point]-0.025,df1['COP'].iloc[optimal_point]-0.1),
            arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0)
            )
 
plt.ylim(3.59,4.25)
plt.xlim(0.395,0.436)
plt.xticks([0.395, 0.405, 0.415, 0.425,0.435],
           [r'0.395', r'0.405', r'0.415', r'0.425',r'0.435'])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'UCH [\$ kWh$^{-1}$]')
plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig12b')
plt.show()
plt.close()



# # #########################
# # ##### Figure 12c #######
# # #########################
# #import data from excel file
# df1 = pd.read_excel('pareto.xlsx',sheet_name='R-410A',header=0) #file name
# #row of optimal value in excel file for
# optimal_point = 53
# #draw dashed box
# x=np.array([df1['UCH'].iloc[0],df1['UCH'].iloc[0],df1['UCH'].iloc[-1],df1['UCH'].iloc[-1],df1['UCH'].iloc[0]])
# y=np.array([df1['COP'].iloc[0],df1['COP'].iloc[-1],df1['COP'].iloc[-1],df1['COP'].iloc[0],df1['COP'].iloc[0]])
# plt.plot(x,y,'--k')
# #plot data point
# plt.plot(df1['UCH'],df1['COP'],'g^',label=r'R-410A')
# #plot optimal TOPSIS point
# plt.plot(df1['UCH'][optimal_point],df1['COP'][optimal_point],'k^',label=r'R-410A')
# #annotation
# plt.annotate('A',xy=(0,0),xytext=(df1['UCH'].iloc[0]-0.002,df1['COP'].iloc[0]-0.0075),annotation_clip=False)
# plt.annotate('B',xy=(0,0),xytext=(df1['UCH'].iloc[-1]+0.0005,df1['COP'].iloc[-1]-0.0075),annotation_clip=False)
# plt.annotate('AB',xy=(0,0),xytext=(df1['UCH'].iloc[optimal_point]-0.003,df1['COP'].iloc[optimal_point]-0.0075),annotation_clip=False)
# 
# plt.annotate('Ideal Pareto solution',xy=(0,0),xytext=(df1['UCH'].iloc[0]+0.0005,df1['COP'].iloc[-1]-0.03),annotation_clip=False)
# plt.annotate('Non-ideal Pareto solution',xy=(0,0),xytext=(df1['UCH'].iloc[-1]-0.022,df1['COP'].iloc[0]+0.01),annotation_clip=False)
# 
# plt.annotate('Ideal point for\nminimum UCH', xy=(df1['UCH'].iloc[0],df1['COP'].iloc[0]), xytext=(df1['UCH'].iloc[0]+0.01,df1['COP'].iloc[0]+0.1),
#             arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0)
#             )
# plt.annotate('Ideal point for\nmaximum $\mathrm{COP_H}$', xy=(df1['UCH'].iloc[-1],df1['COP'].iloc[-1]), xytext=(df1['UCH'].iloc[-1]-0.015,df1['COP'].iloc[-1]-0.15),
#             arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0)
#             )
# plt.annotate('Optimal point using\nTOPSIS method', xy=(df1['UCH'].iloc[optimal_point],df1['COP'].iloc[optimal_point]), xytext=(df1['UCH'].iloc[optimal_point]-0.004,df1['COP'].iloc[optimal_point]-0.2),
#             arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0)
#             )
# 
# plt.ylim(3.38,4.07)
# plt.xlim(0.288,0.331)
# # plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# #            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# # plt.tick_params(
# #     axis='x',          # changes apply to the x-axis
# #     which='both',      # both major and minor ticks are affected
# #     bottom='on',      # ticks along the bottom edge are on
# #     top='off',         # ticks along the top edge are off
# #     labelbottom='on') # labels along the bottom edge are off
# plt.xlabel(r'UCH [\$ kWh$^{-1}$]')
# plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
# # leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# # frame  = leg.get_frame()  
# # frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig12c')
# plt.show()
# plt.close()


# # #########################
# # ##### Figure 12d #######
# # #########################
# #import data from excel file
# df1 = pd.read_excel('pareto.xlsx',sheet_name='R-454A',header=0) #file name
# #row of optimal value in excel file for
# optimal_point = 65
# #draw dashed box
# x=np.array([df1['UCH'].iloc[0],df1['UCH'].iloc[0],df1['UCH'].iloc[-1],df1['UCH'].iloc[-1],df1['UCH'].iloc[0]])
# y=np.array([df1['COP'].iloc[0],df1['COP'].iloc[-1],df1['COP'].iloc[-1],df1['COP'].iloc[0],df1['COP'].iloc[0]])
# plt.plot(x,y,'--k')
# #plot data point
# plt.plot(df1['UCH'],df1['COP'],'yd',label=r'R-454A')
# #plot optimal TOPSIS point
# plt.plot(df1['UCH'][optimal_point],df1['COP'][optimal_point],'kd',label=r'R-454A')
# #annotation
# plt.annotate('A',xy=(0,0),xytext=(df1['UCH'].iloc[0]-0.0014,df1['COP'].iloc[0]-0.0075),annotation_clip=False)
# plt.annotate('B',xy=(0,0),xytext=(df1['UCH'].iloc[-1]+0.0005,df1['COP'].iloc[-1]-0.0075),annotation_clip=False)
# plt.annotate('AB',xy=(0,0),xytext=(df1['UCH'].iloc[optimal_point]-0.002,df1['COP'].iloc[optimal_point]+0.005),annotation_clip=False)
#  
# plt.annotate('Ideal Pareto solution',xy=(0,0),xytext=(df1['UCH'].iloc[0]+0.0005,df1['COP'].iloc[-1]-0.03),annotation_clip=False)
# plt.annotate('Non-ideal Pareto solution',xy=(0,0),xytext=(df1['UCH'].iloc[-1]-0.0155,df1['COP'].iloc[0]+0.01),annotation_clip=False)
#  
# plt.annotate('Ideal point for\nminimum UCH', xy=(df1['UCH'].iloc[0],df1['COP'].iloc[0]), xytext=(df1['UCH'].iloc[0]+0.006,df1['COP'].iloc[0]+0.1),
#             arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0)
#             )
# plt.annotate('Ideal point for\nmaximum $\mathrm{COP_H}$', xy=(df1['UCH'].iloc[-1],df1['COP'].iloc[-1]), xytext=(df1['UCH'].iloc[-1]-0.01,df1['COP'].iloc[-1]-0.35),
#             arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0)
#             )
# plt.annotate('Optimal point using\nTOPSIS method', xy=(df1['UCH'].iloc[optimal_point],df1['COP'].iloc[optimal_point]), xytext=(df1['UCH'].iloc[optimal_point]-0.02,df1['COP'].iloc[optimal_point]-0.15),
#             arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0)
#             )
#  
# plt.ylim(3.8,4.57)
# plt.xlim(0.3255,0.355)
# # plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# #            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# # plt.tick_params(
# #     axis='x',          # changes apply to the x-axis
# #     which='both',      # both major and minor ticks are affected
# #     bottom='on',      # ticks along the bottom edge are on
# #     top='off',         # ticks along the top edge are off
# #     labelbottom='on') # labels along the bottom edge are off
# plt.xlabel(r'UCH [\$ kWh$^{-1}$]')
# plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
# # leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# # frame  = leg.get_frame()  
# # frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig12d')
# plt.show()
# plt.close()


# # #########################
# # ##### Figure 12e #######
# # #########################
# #import data from excel file
# df1 = pd.read_excel('pareto.xlsx',sheet_name='R-452B',header=0) #file name
# #row of optimal value in excel file for
# optimal_point = 32
# #draw dashed box
# x=np.array([df1['UCH'].iloc[0],df1['UCH'].iloc[0],df1['UCH'].iloc[-1],df1['UCH'].iloc[-1],df1['UCH'].iloc[0]])
# y=np.array([df1['COP'].iloc[0],df1['COP'].iloc[-1],df1['COP'].iloc[-1],df1['COP'].iloc[0],df1['COP'].iloc[0]])
# plt.plot(x,y,'--k')
# #plot data point
# plt.plot(df1['UCH'],df1['COP'],'kP',label=r'R-452B')
# #plot optimal TOPSIS point
# plt.plot(df1['UCH'][optimal_point],df1['COP'][optimal_point],'rP',label=r'R-452B')
# #annotation
# plt.annotate('A',xy=(0,0),xytext=(df1['UCH'].iloc[0]-0.0015,df1['COP'].iloc[0]-0.0075),annotation_clip=False)
# plt.annotate('B',xy=(0,0),xytext=(df1['UCH'].iloc[-1]+0.0005,df1['COP'].iloc[-1]-0.0075),annotation_clip=False)
# plt.annotate('AB',xy=(0,0),xytext=(df1['UCH'].iloc[optimal_point]-0.002,df1['COP'].iloc[optimal_point]+0.01),annotation_clip=False)
#  
# plt.annotate('Ideal Pareto solution',xy=(0,0),xytext=(df1['UCH'].iloc[0]+0.0005,df1['COP'].iloc[-1]-0.03),annotation_clip=False)
# plt.annotate('Non-ideal Pareto solution',xy=(0,0),xytext=(df1['UCH'].iloc[-1]-0.018,df1['COP'].iloc[0]+0.01),annotation_clip=False)
#  
# plt.annotate('Ideal point for\nminimum UCH', xy=(df1['UCH'].iloc[0],df1['COP'].iloc[0]), xytext=(df1['UCH'].iloc[0]+0.008,df1['COP'].iloc[0]+0.1),
#             arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0)
#             )
# plt.annotate('Ideal point for\nmaximum $\mathrm{COP_H}$', xy=(df1['UCH'].iloc[-1],df1['COP'].iloc[-1]), xytext=(df1['UCH'].iloc[-1]-0.012,df1['COP'].iloc[-1]-0.22),
#             arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0)
#             )
# plt.annotate('Optimal point using\nTOPSIS method', xy=(df1['UCH'].iloc[optimal_point],df1['COP'].iloc[optimal_point]), xytext=(df1['UCH'].iloc[optimal_point]-0.017,df1['COP'].iloc[optimal_point]-0.07),
#             arrowprops=dict(facecolor='black',arrowstyle='-|>',shrinkB=2.75,lw=1.0)
#             )
#  
# plt.ylim(3.45,4.11)
# plt.xlim(0.2575,0.294)
# # plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# #            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# # plt.tick_params(
# #     axis='x',          # changes apply to the x-axis
# #     which='both',      # both major and minor ticks are affected
# #     bottom='on',      # ticks along the bottom edge are on
# #     top='off',         # ticks along the top edge are off
# #     labelbottom='on') # labels along the bottom edge are off
# plt.xlabel(r'UCH [\$ kWh$^{-1}$]')
# plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
# # leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# # frame  = leg.get_frame()  
# # frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig12e')
# plt.show()
# plt.close()
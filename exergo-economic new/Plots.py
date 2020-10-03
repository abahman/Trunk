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
##### T_evap #######
#########################

#import data from excel file
df = pd.read_excel('Plots.xlsx',sheet_name='T_evap',header=0) #file name


fig, ax = plt.subplots(figsize=(6,9))
plt.subplot(3, 1, 1)
plt.plot(df['X'],df['COP_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['COP_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['COP_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['COP_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['COP_R452B'],'P-',color='k',label=r'R-452B')   
   
     
# plt.ylim(2.6,4)
plt.xlim(-36,-24)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are on
    top='on',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
# plt.xlabel(r'$T_{evap}$ [$\degree$C]')
plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig4a')
# plt.show()
# plt.close()
   
plt.subplot(3, 1, 2)
plt.plot(df['X'],df['eta_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['eta_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['eta_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['eta_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['eta_R452B'],'P-',color='k',label=r'R-452B')   
   
     
plt.ylim(39,65)
plt.xlim(-36,-24)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are on
    top='on',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
# plt.xlabel(r'$T_{evap}$ [$\degree$C]')
plt.ylabel(r'$\eta_{ex}$ [%]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig4b')
# plt.show()
# plt.close()
   
 
plt.subplot(3, 1, 3)
plt.plot(df['X'],df['UCH_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['UCH_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['UCH_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['UCH_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['UCH_R452B'],'P-',color='k',label=r'R-452B')   
   
     
# plt.ylim(100,600)
plt.xlim(-36,-24)
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
# leg = plt.legend(fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
# frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('T_evap')
plt.show()
plt.close()



#########################
##### T_cond #######
#########################

#import data from excel file
df = pd.read_excel('Plots.xlsx',sheet_name='T_cond',header=0) #file name


fig, ax = plt.subplots(figsize=(6,9))
plt.subplot(3, 1, 1)
plt.plot(df['X'],df['COP_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['COP_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['COP_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['COP_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['COP_R452B'],'P-',color='k',label=r'R-452B')  
  
    
plt.ylim(2.0,4.6)
plt.xlim(54, 65)
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
plt.plot(df['X'],df['eta_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['eta_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['eta_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['eta_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['eta_R452B'],'P-',color='k',label=r'R-452B')     
  
    
# plt.ylim(2.0,4.55)
plt.xlim(54, 65)
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
plt.plot(df['X'],df['UCH_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['UCH_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['UCH_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['UCH_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['UCH_R452B'],'P-',color='k',label=r'R-452B')    
  
    
# plt.ylim(0.2,0.5)
plt.xlim(54, 65)
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
savefigs('T_cond')
plt.show()
plt.close()



#########################
##### Sh_evap #######
#########################
#import data from excel file
df = pd.read_excel('Plots.xlsx',sheet_name='Sh_evap',header=0) #file name


fig, ax = plt.subplots(figsize=(6,9))
plt.subplot(3, 1, 1)
plt.plot(df['X'],df['COP_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['COP_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['COP_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['COP_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['COP_R452B'],'P-',color='k',label=r'R-452B')  
    
      
# plt.ylim(100,600)
plt.xlim(2.5,8.5)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are on
    top='on',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
# plt.xlabel(r'$Sh_{evap}$ [$\degree$C]')
plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig6a')
# plt.show()
# plt.close()
 
 
 
plt.subplot(3, 1, 2) 
plt.plot(df['X'],df['eta_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['eta_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['eta_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['eta_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['eta_R452B'],'P-',color='k',label=r'R-452B')     
    
      
plt.ylim(40,61)
plt.xlim(2.5,8.5)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are on
    top='on',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
# plt.xlabel(r'$Sh_{evap}$ [$\degree$C]')
plt.ylabel(r'$\eta_{ex}$ [%]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig6b')
# plt.show()
# plt.close()
    


plt.subplot(3, 1, 3)
plt.plot(df['X'],df['UCH_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['UCH_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['UCH_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['UCH_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['UCH_R452B'],'P-',color='k',label=r'R-452B')     
    
      
plt.ylim(0.2,0.265)
plt.xlim(2.5,8.5)
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
# leg = plt.legend(fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('Sh_evap')
plt.show()
plt.close()





#########################
##### Sc_cond #######
#########################
#import data from excel file
df = pd.read_excel('Plots.xlsx',sheet_name='Sc_cond',header=0) #file name


fig, ax = plt.subplots(figsize=(6,9))
plt.subplot(3, 1, 1)
plt.plot(df['X'],df['COP_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['COP_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['COP_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['COP_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['COP_R452B'],'P-',color='k',label=r'R-452B')    
   
     
# plt.ylim(2.7,3.4)
plt.xlim(2.5,8.5)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are on
    top='on',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
# plt.xlabel(r'$Sc_{cond}$ [$\degree$C]')
plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig7a')
# plt.show()
# plt.close()
   


plt.subplot(3, 1, 2) 
plt.plot(df['X'],df['eta_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['eta_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['eta_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['eta_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['eta_R452B'],'P-',color='k',label=r'R-452B')   
   
     
# plt.ylim(100,600)
plt.xlim(2.5,8.5)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are on
    top='on',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
# plt.xlabel(r'$Sc_{cond}$ [$\degree$C]')
plt.ylabel(r'$\eta_{ex}$ [%]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig7b')
# plt.show()
# plt.close()
   


plt.subplot(3, 1, 3) 
plt.plot(df['X'],df['UCH_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['UCH_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['UCH_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['UCH_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['UCH_R452B'],'P-',color='k',label=r'R-452B') 
   
     
# plt.ylim(100,600)
plt.xlim(2.5,8.5)
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
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('Sc_cond')
plt.show()
plt.close()





#########################
##### TTD #######
#########################
#import data from excel file
df = pd.read_excel('Plots.xlsx',sheet_name='TTD',header=0) #file name


fig, ax = plt.subplots(figsize=(6,9))
plt.subplot(3, 1, 1)
plt.plot(df['X'],df['COP_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['COP_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['COP_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['COP_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['COP_R452B'],'P-',color='k',label=r'R-452B')
  
    
# plt.ylim(100,600)
plt.xlim(7,13)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are on
    top='on',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
# plt.xlabel(r'$TTD$ [$\degree$C]')
plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig8a')
# plt.show()
# plt.close()
  

plt.subplot(3, 1, 2)
plt.plot(df['X'],df['eta_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['eta_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['eta_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['eta_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['eta_R452B'],'P-',color='k',label=r'R-452B')    
  
    
plt.ylim(39,65)
plt.xlim(7,13)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are on
    top='on',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
# plt.xlabel(r'$TTD$ [$\degree$C]')
plt.ylabel(r'$\eta_{ex}$ [%]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig8b')
# plt.show()
# plt.close()
#   
  

plt.subplot(3, 1, 3)
plt.plot(df['X'],df['UCH_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['UCH_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['UCH_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['UCH_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['UCH_R452B'],'P-',color='k',label=r'R-452B') 
  
    
# plt.ylim(100,600)
plt.xlim(7,13)
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
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('TTD')
plt.show()
plt.close()




#########################
##### Sh_econ #######
#########################
#import data from excel file
df = pd.read_excel('Plots.xlsx',sheet_name='Sh_econ',header=0) #file name


fig, ax = plt.subplots(figsize=(6,9))
plt.subplot(3, 1, 1)
plt.plot(df['X'],df['COP_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['COP_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['COP_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['COP_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['COP_R452B'],'P-',color='k',label=r'R-452B')
    
      
plt.ylim(2.2,3.9)
plt.xlim(2.5,8.5)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are on
    top='on',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
# plt.xlabel(r'$Sh_{econ}$ [$\degree$C]')
plt.ylabel(r'$\mathrm{COP_{H}}$ [$-$]') #{\textdegree}C
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig11a')
# plt.show()
# plt.close()


plt.subplot(3, 1, 2)
plt.plot(df['X'],df['eta_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['eta_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['eta_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['eta_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['eta_R452B'],'P-',color='k',label=r'R-452B')    
    
      
# plt.ylim(100,600)
plt.xlim(2.5,8.5)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are on
    top='on',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
# plt.xlabel(r'$Sh_{econ}$ [$\degree$C]')
plt.ylabel(r'$\eta_{ex}$ [%]') #{\textdegree}C
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig11b')
# plt.show()
# plt.close()
#     
    

plt.subplot(3, 1, 3)
plt.plot(df['X'],df['UCH_R32'],'ro-',label=r'R-32')    
plt.plot(df['X'],df['UCH_R290'],'bs-',label=r'R-290')  
plt.plot(df['X'],df['UCH_R410A'],'g^-',label=r'R-410A')  
plt.plot(df['X'],df['UCH_R454A'],'yd-',label=r'R-454A')  
plt.plot(df['X'],df['UCH_R452B'],'P-',color='k',label=r'R-452B')  
    
      
plt.ylim(0.2,0.265)
plt.xlim(2.5,8.5)
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
# leg = plt.legend(fancybox=False,numpoints=1,loc='upper left',bbox_to_anchor=(1.0,1.025))
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('Sh_econ')
plt.show()
plt.close()
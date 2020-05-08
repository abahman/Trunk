'''
Created on Jan 17, 2017

@author: ammarbahman

Note: this file plots the Ts/Ph diagrams  in VI cycle ECU for IJR paper

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CoolProp
from CoolProp.Plots import PropertyPlot
import matplotlib as mpl
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

#Baseline data (no injection)
df18 = pd.read_excel('data_tables.xlsx',sheet_name='18K ECU',header=20)
df36 = pd.read_excel('data_tables.xlsx',sheet_name='36K ECU',header=20)
df60 = pd.read_excel('data_tables.xlsx',sheet_name='60K ECU',header=20)

P18 = np.array(df18[1:13]["P[i]"])
P36 = np.array(df36[1:13]["P[i]"])
P60 = np.array(df60[1:13]["P[i]"])
T18 = np.array(df18[1:13]["T[i]"])
T36 = np.array(df36[1:13]["T[i]"])
T60 = np.array(df60[1:13]["T[i]"])
h18 = np.array(df18[1:13]["h[i]"])
h36 = np.array(df36[1:13]["h[i]"])
h60 = np.array(df60[1:13]["h[i]"])
s18 = np.array(df18[1:13]["s[i]"])
s36 = np.array(df36[1:13]["s[i]"])
s60 = np.array(df60[1:13]["s[i]"])
# B_T = np.append(np.array(df[1:8]["T[i]"]),np.array(df[9:13]["T[i]"]))#using append to remove unnecessary points


df = pd.read_excel('data_tables.xlsx',sheet_name='x=0(R407C)',header=0)
Px0_407C = np.array(df[1:]["p"])
hx0_407C = np.array(df[1:]["h"])
Tx0_407C = np.array(df[1:]["T"])
sx0_407C = np.array(df[1:]["s"])
df = pd.read_excel('data_tables.xlsx',sheet_name='x=1(R407C)',header=0)
Px1_407C = np.array(df[1:]["p"])
hx1_407C = np.array(df[1:]["h"])
Tx1_407C = np.array(df[1:]["T"])
sx1_407C = np.array(df[1:]["s"])

df = pd.read_excel('data_tables.xlsx',sheet_name='x=0(R410A)',header=0)
Px0_410A = np.array(df[1:]["p"])
hx0_410A = np.array(df[1:]["h"])
Tx0_410A = np.array(df[1:]["T"])
sx0_410A = np.array(df[1:]["s"])
df = pd.read_excel('data_tables.xlsx',sheet_name='x=1(R410A)',header=0)
Px1_410A = np.array(df[1:]["p"])
hx1_410A = np.array(df[1:]["h"])
Tx1_410A = np.array(df[1:]["T"])
sx1_410A = np.array(df[1:]["s"])

#plotting           
ref_fluid = 'HEOS::R407C'
####################################
#Plot P-h diagram  
####################################
ph_plot_R407C = PropertyPlot(ref_fluid,'Ph')
# ph_plot_R407C.calc_isolines(CoolProp.iQ, num=2)
# ph_plot_R407C.props[CoolProp.iQ]['lw'] = 0.5
ph_plot_R407C.draw() #actually draw isoline
ph_plot_R407C.isolines.clear() #stop isoline, to avoid ploting the isoline at the end 
ph_plot_R407C.title('P-h R407C')
ph_plot_R407C.xlabel(r'$h$ [kJ/kg]')#r'$h$ [{kJ} {kg$^{-1}$}]'
ph_plot_R407C.ylabel(r'$P$ [kPa]')
ph_plot_R407C.axis.set_yscale('log')

maxline, = plt.plot(hx0_407C,Px0_407C,'k-',linewidth=1.25)
plt.plot(hx1_407C,Px1_407C,'k-',linewidth=1.25)
minline, = plt.plot(hx0_410A,Px0_410A,'b--',linewidth=1.25) 
plt.plot(hx1_410A,Px1_410A,'b--',linewidth=1.25) 

plt.plot(h60,P60,'r-',linewidth=1.0,label='5-RT ECU') 
plt.plot(h36,P36,'b--',linewidth=1.0,label='3-RT ECU')
plt.plot(h18,P18,'g:',linewidth=1.0,label='1.5-RT ECU')
 
plt.ylim(500,6500)
plt.xlim(200,500)
plt.title('')
plt.yticks([500, 1000, 2000, 5000],
           [r'$500$', r'$1000$', r'$2000$', r'$5000$'])
# Add first legend:  only labeled data is included
leg1=plt.legend(loc='upper left',fancybox=False,numpoints=1)
# Add second legend for the maxes and mins.
# leg1 will be removed from figure
leg2=plt.legend([minline,maxline],['R-410A','R-407C'],loc='lower left',frameon=False,numpoints=1,fontsize=8)
# Manually add the first legend back
ph_plot_R407C.axis.add_artist(leg1)
frame=leg1.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
ph_plot_R407C.savefig('p-h.pdf')    
ph_plot_R407C.show()
plt.close()

####################################
#Plot T-s diagram  
####################################
ts_plot_R407C = PropertyPlot(ref_fluid, 'Ts',unit_system='KSI')
# ts_plot_R407C.calc_isolines(CoolProp.iQ, num=2)
ts_plot_R407C.draw() #actually draw isoline
ts_plot_R407C.isolines.clear() #stop isoline, to avoid ploting the isoline at the end 
ts_plot_R407C.title('T-s R407C')
ts_plot_R407C.xlabel(r'$s$ [kJ/kg-K]')#r'$s$ [{kJ} {kg$^{-1}$ K$^{-1}$}]'
ts_plot_R407C.ylabel(r'$T$ [$\degree$C]')#{\textdegree}C
#ts_plot_R407C.grid()
 
maxline, = plt.plot(sx0_407C,Tx0_407C,'k-',linewidth=1.25)
plt.plot(sx1_407C,Tx1_407C,'k-',linewidth=1.25)
minline, = plt.plot(sx0_410A,Tx0_410A,'b--',linewidth=1.25) 
plt.plot(sx1_410A,Tx1_410A,'b--',linewidth=1.25) 

plt.plot(s60,T60,'r-',linewidth=1.0,label='5-RT ECU') 
plt.plot(s36,T36,'b--',linewidth=1.0,label='3-RT ECU')
plt.plot(s18,T18,'g:',linewidth=1.0,label='1.5-RT ECU')


plt.ylim([-10,120])
plt.xlim([1.0,2.0])
plt.title('')
# Add first legend:  only labeled data is included
leg1=plt.legend(loc='upper left',fancybox=False,numpoints=1)
# Add second legend for the maxes and mins.
# leg1 will be removed from figure
leg2=plt.legend([minline,maxline],['R-410A','R-407C'],loc='lower left',frameon=False,numpoints=1,fontsize=8)
# Manually add the first legend back
ts_plot_R407C.axis.add_artist(leg1)
frame=leg1.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
ts_plot_R407C.savefig('T-s.pdf')    
ts_plot_R407C.show()
plt.close()
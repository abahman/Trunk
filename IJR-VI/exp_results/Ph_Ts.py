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
df = pd.read_excel("Ts_Ph_Test1_VI.xlsx",sheetname='no injection')
B_P = np.append(np.array(df[1:8]["P[i]"]),np.array(df[9:13]["P[i]"])) #using append to remove unnecessary points
B_T = np.append(np.array(df[1:8]["T[i]"]),np.array(df[9:13]["T[i]"]))
B_h = np.append(np.array(df[1:8]["h[i]"]),np.array(df[9:13]["h[i]"]))
B_s = np.append(np.array(df[1:8]["s[i]"]),np.array(df[9:13]["s[i]"]))

#7K superheat data
df = pd.read_excel("Ts_Ph_Test1_VI.xlsx",sheetname='7K')
VI_7K_P = np.append(np.append(np.array(df[2:6]["P[i]"]),df.iloc[7]["P[i]"]),np.array(df[9:13]["P[i]"]))
VI_7K_T = np.append(np.append(np.array(df[2:6]["T[i]"]),df.iloc[7]["T[i]"]),np.array(df[9:13]["T[i]"]))
VI_7K_h = np.append(np.append(np.array(df[2:6]["h[i]"]),df.iloc[7]["h[i]"]),np.array(df[9:13]["h[i]"]))
VI_7K_s = np.append(np.append(np.array(df[2:6]["s[i]"]),df.iloc[7]["s[i]"]),np.array(df[9:13]["s[i]"]))
VI_7K_P1 = np.append(np.array(df[103:106]["P[i]"]),df.iloc[302]["P[i]"])
VI_7K_T1 = np.append(np.array(df[103:106]["T[i]"]),df.iloc[302]["T[i]"])
VI_7K_h1 = np.append(np.array(df[103:106]["h[i]"]),df.iloc[302]["h[i]"])
VI_7K_s1 = np.append(np.array(df[103:106]["s[i]"]),df.iloc[302]["s[i]"])
VI_7K_P2 = np.array(df[300:304]["P[i]"])
VI_7K_T2 = np.array(df[300:304]["T[i]"])
VI_7K_h2 = np.array(df[300:304]["h[i]"])
VI_7K_s2 = np.array(df[300:304]["s[i]"])

#0K superheat data
df = pd.read_excel("Ts_Ph_Test1_VI.xlsx",sheetname='0K')
VI_0K_P = np.append(np.append(np.array(df[2:6]["P[i]"]),df.iloc[7]["P[i]"]),np.array(df[9:13]["P[i]"]))
VI_0K_T = np.append(np.append(np.array(df[2:6]["T[i]"]),df.iloc[7]["T[i]"]),np.array(df[9:13]["T[i]"]))
VI_0K_h = np.append(np.append(np.array(df[2:6]["h[i]"]),df.iloc[7]["h[i]"]),np.array(df[9:13]["h[i]"]))
VI_0K_s = np.append(np.append(np.array(df[2:6]["s[i]"]),df.iloc[7]["s[i]"]),np.array(df[9:13]["s[i]"]))
VI_0K_P1 = np.append(np.array(df[103:105]["P[i]"]),df.iloc[302]["P[i]"])
VI_0K_T1 = np.append(np.array(df[103:105]["T[i]"]),df.iloc[302]["T[i]"])
VI_0K_h1 = np.append(np.array(df[103:105]["h[i]"]),df.iloc[302]["h[i]"])
VI_0K_s1 = np.append(np.array(df[103:105]["s[i]"]),df.iloc[302]["s[i]"])
VI_0K_P2 = np.array(df[300:304]["P[i]"])
VI_0K_T2 = np.array(df[300:304]["T[i]"])
VI_0K_h2 = np.array(df[300:304]["h[i]"])
VI_0K_s2 = np.array(df[300:304]["s[i]"])

#constant quality lines data
df = pd.read_excel("Ts_Ph_Test1_VI.xlsx",sheetname="x=0.8")
Px8 = np.array(df[:]["p"])
hx8 = np.array(df[:]["h"])
Tx8 = np.array(df[:]["T"])
sx8 = np.array(df[:]["s"])

df = pd.read_excel("Ts_Ph_Test1_VI.xlsx",sheetname="x=0.6")
Px6 = np.array(df[:]["p"])
hx6 = np.array(df[:]["h"])
Tx6 = np.array(df[:]["T"])
sx6 = np.array(df[:]["s"])

df = pd.read_excel("Ts_Ph_Test1_VI.xlsx",sheetname="x=0.4")
Px4 = np.array(df[:]["p"])
hx4 = np.array(df[:]["h"])
Tx4 = np.array(df[:]["T"])
sx4 = np.array(df[:]["s"])

df = pd.read_excel("Ts_Ph_Test1_VI.xlsx",sheetname="x=0.2")
Px2 = np.array(df[:]["p"])
hx2 = np.array(df[:]["h"])
Tx2 = np.array(df[:]["T"])
sx2 = np.array(df[:]["s"])

df = pd.read_excel("Ts_Ph_Test1_VI.xlsx",sheetname="x=0")
Tx0 = np.array(df[:]["T"])
sx0 = np.array(df[:]["s"])

df = pd.read_excel("Ts_Ph_Test1_VI.xlsx",sheetname="x=1")
Tx1 = np.array(df[:]["T"])
sx1 = np.array(df[:]["s"])

#plotting           
ref_fluid = 'HEOS::R407C'
#Plot P-h diagram 
ph_plot_R407C = PropertyPlot(ref_fluid, 'Ph')
ph_plot_R407C.calc_isolines(CoolProp.iQ, num=2)
ph_plot_R407C.draw() #actually draw isoline
ph_plot_R407C.isolines.clear() #stop isoline, to avoid ploting the isoline at the end 
ph_plot_R407C.title('P-h R407C')
ph_plot_R407C.xlabel(r'$h$ [kJ kg$^{-1}$]')#r'$h$ [{kJ} {kg$^{-1}$}]'
ph_plot_R407C.ylabel(r'$P$ [kPa]')
ph_plot_R407C.axis.set_yscale('log')
#ph_plot_R407C.grid()
plt.plot(hx8,Px8,color="grey",linewidth=0.25)
plt.text(364, 600, '$x$=0.8',color="grey",fontsize=5,rotation=74)
plt.plot(hx6,Px6,color="grey",linewidth=0.25)
plt.text(322, 600, '$x$=0.6',color="grey",fontsize=5,rotation=70)
plt.plot(hx4,Px4,color="grey",linewidth=0.25)
plt.text(279, 600, '$x$=0.4',color="grey",fontsize=5,rotation=68)
plt.plot(hx2,Px2,color="grey",linewidth=0.25)
plt.text(236, 600, '$x$=0.2',color="grey",fontsize=5,rotation=67)
 
plt.plot(B_h,B_P,'-*',linewidth=1.5,alpha=0.9,label='Baseline')
plt.plot(VI_7K_h,VI_7K_P,'r--X',linewidth=1.5,alpha=0.9, label='Superheated')
plt.plot(VI_7K_h1,VI_7K_P1,'r--X',linewidth=1.5,alpha=0.9)
plt.plot(VI_7K_h2,VI_7K_P2,'r--X',linewidth=1.5,alpha=0.9)
plt.plot(VI_0K_h,VI_0K_P,'g:P',linewidth=1.5,alpha=0.9,label='Saturated')
plt.plot(VI_0K_h1,VI_0K_P1,'g:P',linewidth=1.5,alpha=0.9)
plt.plot(VI_0K_h2,VI_0K_P2,'g:P',linewidth=1.5,alpha=0.9)
 
#states numbers
plt.text(428, 750, '1',fontsize=8)
plt.text(482, 3400, '2',fontsize=8)
plt.text(290, 3400, '3',fontsize=8)
plt.text(245, 3000, '4',fontsize=8)
plt.text(245, 1500, '5',fontsize=8)
plt.text(245, 750, '6',fontsize=8)
plt.text(430, 1500, '7',fontsize=8)
 
 
plt.ylim(500,5000)
plt.xlim(200,500)
plt.title('')
plt.yticks([500, 1000, 2000, 5000],
           [r'$500$', r'$1000$', r'$2000$', r'$5000$'])
plt.text(475,600,'R-407C',ha='center',va='top')
leg=plt.legend(loc='upper left',fancybox=False,numpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout() 
ph_plot_R407C.savefig('p-h.pdf')    
ph_plot_R407C.show()
plt.close()

#Plot T-s diagram  
ts_plot_R407C = PropertyPlot(ref_fluid, 'Ts',unit_system='KSI')
ts_plot_R407C.calc_isolines(CoolProp.iQ, num=2)
ts_plot_R407C.draw() #actually draw isoline
ts_plot_R407C.isolines.clear() #stop isoline, to avoid ploting the isoline at the end 
ts_plot_R407C.title('T-s R407C')
ts_plot_R407C.xlabel(r'$s$ [kJ kg$^{-1}$ K$^{-1}$]')#r'$s$ [{kJ} {kg$^{-1}$ K$^{-1}$}]'
ts_plot_R407C.ylabel(r'$T$ [$\degree$C]')#{\textdegree}C
#ts_plot_R407C.grid()
plt.plot(sx0,Tx0,color="k",linewidth=0.6)
plt.plot(sx1,Tx1,color="k",linewidth=0.6)
plt.plot(sx8,Tx8,color="grey",linewidth=0.25)
plt.text(1.604, -5, '$x$=0.8',color="grey",fontsize=5,rotation=90)
plt.plot(sx6,Tx6,color="grey",linewidth=0.25)
plt.text(1.431, -5, '$x$=0.6',color="grey",fontsize=5,rotation=71)
plt.plot(sx4,Tx4,color="grey",linewidth=0.25)
plt.text(1.262, -5, '$x$=0.4',color="grey",fontsize=5,rotation=58)
plt.plot(sx2,Tx2,color="grey",linewidth=0.25)
plt.text(1.095, -5, '$x$=0.2',color="grey",fontsize=5,rotation=48)
 
plt.plot(B_s,B_T,'-*',linewidth=1.5,alpha=0.9,label='Baseline')
plt.plot(VI_7K_s,VI_7K_T,'r--X',linewidth=1.5,alpha=0.9, label='Superheated')
plt.plot(VI_7K_s1,VI_7K_T1,'r--X',linewidth=1.5,alpha=0.9)
plt.plot(VI_7K_s2,VI_7K_T2,'r--X',linewidth=1.5,alpha=0.9)
plt.plot(VI_0K_s,VI_0K_T,'g:P',linewidth=1.5,alpha=0.9,label='Saturated')
plt.plot(VI_0K_s1,VI_0K_T1,'g:P',linewidth=1.5,alpha=0.9)
plt.plot(VI_0K_s2,VI_0K_T2,'g:P',linewidth=1.5,alpha=0.9)

#states numbers
plt.text(1.8, 20, '1',fontsize=8)
plt.text(1.85, 106, '2',fontsize=8)
plt.text(1.28, 60, '3',fontsize=8)
plt.text(1.15, 38, '4',fontsize=8)
plt.text(1.15, 25, '5',fontsize=8)
plt.text(1.16, 7, '6',fontsize=8)
plt.text(1.75, 48, '7',fontsize=8)

plt.ylim([-20,120])
plt.xlim([1.0,2.0])
plt.title('')
plt.text(1.90,-5,'R-407C',ha='center',va='top')
leg=plt.legend(loc='upper left',fancybox=False, numpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout() 
ts_plot_R407C.savefig('T-s.pdf')    
ts_plot_R407C.show()
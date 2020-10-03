'''
Created on Jun 12, 2020

@author: ammarbahman

Note: this file plots the Ts/Ph diagrams  in exergo-economic paper

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

def savefigs(name):
    #plt.savefig(name+'.eps')
    plt.savefig('images/'+name+'.pdf')
    #plt.savefig(name+'.png',dpi=600)
    #plt.show()
    
#Baseline data (no injection)
df = pd.read_excel("Ts_Ph.xlsx",sheet_name='states',header=0)
P = np.array(df['P[i]'])   #np.append(np.array(df[1:8]["P[i]"]),np.array(df[9:13]["P[i]"])) #using append to remove unnecessary points
T = np.array(df['T[i]'])     #np.append(np.array(df[1:8]["T[i]"]),np.array(df[9:13]["T[i]"]))

from CoolProp.CoolProp import PropsSI
# print(PropsSI('T','P',3430.2*1000,'Q',0,'REFPROP::R410A')-273.15)
h = PropsSI('H','P',P*1000,'T',T+273.15,'REFPROP::R410A')/1000  #[kJ/kg]   #np.append(np.array(df[1:8]["h[i]"]),np.array(df[9:13]["h[i]"]))
s = PropsSI('S','P',P*1000,'T',T+273.15,'REFPROP::R410A')/1000 #[kJ/kg-K] #np.append(np.array(df[1:8]["s[i]"]),np.array(df[9:13]["s[i]"]))


#7K superheat data
# df = pd.read_excel("Ts_Ph_Test1_VI.xlsx",sheet_name='7K')
# VI_7K_P = np.append(np.append(np.array(df[2:6]["P[i]"]),df.iloc[7]["P[i]"]),np.array(df[9:13]["P[i]"]))
# VI_7K_T = np.append(np.append(np.array(df[2:6]["T[i]"]),df.iloc[7]["T[i]"]),np.array(df[9:13]["T[i]"]))
# VI_7K_h = np.append(np.append(np.array(df[2:6]["h[i]"]),df.iloc[7]["h[i]"]),np.array(df[9:13]["h[i]"]))
# VI_7K_s = np.append(np.append(np.array(df[2:6]["s[i]"]),df.iloc[7]["s[i]"]),np.array(df[9:13]["s[i]"]))
# VI_7K_P1 = np.append(np.array(df[103:106]["P[i]"]),df.iloc[302]["P[i]"])
# VI_7K_T1 = np.append(np.array(df[103:106]["T[i]"]),df.iloc[302]["T[i]"])
# VI_7K_h1 = np.append(np.array(df[103:106]["h[i]"]),df.iloc[302]["h[i]"])
# VI_7K_s1 = np.append(np.array(df[103:106]["s[i]"]),df.iloc[302]["s[i]"])
# VI_7K_P2 = np.array(df[300:304]["P[i]"])
# VI_7K_T2 = np.array(df[300:304]["T[i]"])
# VI_7K_h2 = np.array(df[300:304]["h[i]"])
# VI_7K_s2 = np.array(df[300:304]["s[i]"])

#0K superheat data
# df = pd.read_excel("Ts_Ph_Test1_VI.xlsx",sheet_name='0K')
# VI_0K_P = np.append(np.append(np.array(df[2:6]["P[i]"]),df.iloc[7]["P[i]"]),np.array(df[9:13]["P[i]"]))
# VI_0K_T = np.append(np.append(np.array(df[2:6]["T[i]"]),df.iloc[7]["T[i]"]),np.array(df[9:13]["T[i]"]))
# VI_0K_h = np.append(np.append(np.array(df[2:6]["h[i]"]),df.iloc[7]["h[i]"]),np.array(df[9:13]["h[i]"]))
# VI_0K_s = np.append(np.append(np.array(df[2:6]["s[i]"]),df.iloc[7]["s[i]"]),np.array(df[9:13]["s[i]"]))
# VI_0K_P1 = np.append(np.array(df[103:105]["P[i]"]),df.iloc[302]["P[i]"])
# VI_0K_T1 = np.append(np.array(df[103:105]["T[i]"]),df.iloc[302]["T[i]"])
# VI_0K_h1 = np.append(np.array(df[103:105]["h[i]"]),df.iloc[302]["h[i]"])
# VI_0K_s1 = np.append(np.array(df[103:105]["s[i]"]),df.iloc[302]["s[i]"])
# VI_0K_P2 = np.array(df[300:304]["P[i]"])
# VI_0K_T2 = np.array(df[300:304]["T[i]"])
# VI_0K_h2 = np.array(df[300:304]["h[i]"])
# VI_0K_s2 = np.array(df[300:304]["s[i]"])

#constant quality lines data
df = pd.read_excel("Ts_Ph.xlsx",sheet_name="property")
Px8 = np.array(df[:]["p"])
hx8 = PropsSI('H','P',Px8*1000,'Q',0.8,'REFPROP::R410A')/1000  #[kJ/kg]
Tx8 = np.array(df[:]["T"])
sx8 = PropsSI('S','T',Tx8+273.15,'Q',0.8,'REFPROP::R410A')/1000 #[kJ/kg-K]

df = pd.read_excel("Ts_Ph.xlsx",sheet_name="property")
Px6 = np.array(df[:]["p"])
hx6 = PropsSI('H','P',Px6*1000,'Q',0.6,'REFPROP::R410A')/1000  #[kJ/kg]
Tx6 = np.array(df[:]["T"])
sx6 = PropsSI('S','T',Tx6+273.15,'Q',0.6,'REFPROP::R410A')/1000 #[kJ/kg-K]

df = pd.read_excel("Ts_Ph.xlsx",sheet_name="property")
Px4 = np.array(df[:]["p"])
hx4 = PropsSI('H','P',Px4*1000,'Q',0.4,'REFPROP::R410A')/1000  #[kJ/kg]
Tx4 = np.array(df[:]["T"])
sx4 = PropsSI('S','T',Tx4+273.15,'Q',0.4,'REFPROP::R410A')/1000 #[kJ/kg-K]

df = pd.read_excel("Ts_Ph.xlsx",sheet_name="property")
Px2 = np.array(df[:]["p"])
hx2 = PropsSI('H','P',Px2*1000,'Q',0.2,'REFPROP::R410A')/1000  #[kJ/kg]
Tx2 = np.array(df[:]["T"])
sx2 = PropsSI('S','T',Tx2+273.15,'Q',0.2,'REFPROP::R410A')/1000 #[kJ/kg-K]

df = pd.read_excel("Ts_Ph.xlsx",sheet_name="property")
Px0 = np.array(df[:]["p"])
hx0 = PropsSI('H','P',Px0*1000,'Q',0.0,'REFPROP::R410A')/1000 #[kJ/kg]
Tx0 = np.array(df[:]["T"])
sx0 = PropsSI('S','T',Tx0+273.15,'Q',0.0,'REFPROP::R410A')/1000 #[kJ/kg-K]
 
df = pd.read_excel("Ts_Ph.xlsx",sheet_name="property")
Px1 = np.array(df[:]["p"])
hx1 = PropsSI('H','P',Px0*1000,'Q',1.0,'REFPROP::R410A')/1000 #[kJ/kg]
Tx1 = np.array(df[:]["T"])
sx1 = PropsSI('S','T',Tx1+273.15,'Q',1.0,'REFPROP::R410A')/1000 #[kJ/kg-K]

#plotting           
ref_fluid = 'REFPROP::R410A'
######################
#Plot P-h diagram
######################
# ph_plot_R410A = PropertyPlot(ref_fluid, 'Ph')
# ph_plot_R410A.calc_isolines(CoolProp.iQ, num=2)
# ph_plot_R410A.draw() #actually draw isoline
# ph_plot_R410A.isolines.clear() #stop isoline, to avoid ploting the isoline at the end 
# ph_plot_R410A.title('P-h R410A')
# ph_plot_R410A.xlabel(r'$h$ [kJ kg$^{-1}$]')#r'$h$ [{kJ} {kg$^{-1}$}]'
# ph_plot_R410A.ylabel(r'$P$ [kPa]')
# ph_plot_R410A.axis.set_yscale('log')
#ph_plot_R407C.grid()
plt.plot(hx0,Px0,color="k",linewidth=1)
plt.plot(hx1,Px1,color="k",linewidth=1)
plt.plot(hx8,Px8,color="grey",linewidth=0.25)
plt.text(363, 650, '$x$=0.8',color="grey",fontsize=6,rotation=74)
plt.plot(hx6,Px6,color="grey",linewidth=0.25)
plt.text(317, 650, '$x$=0.6',color="grey",fontsize=6,rotation=70)
plt.plot(hx4,Px4,color="grey",linewidth=0.25)
plt.text(270, 650, '$x$=0.4',color="grey",fontsize=6,rotation=68)
plt.plot(hx2,Px2,color="grey",linewidth=0.25)
plt.text(223, 650, '$x$=0.2',color="grey",fontsize=6,rotation=67)
  
h[9] = h[8] # to over write state 9 enthalpy with state 8 because isenthalpic expansion.
enthalpy_1 = np.append(np.append(h[0:7],h[8:11]), h[0])
pressure_1 = np.append(np.append(P[0:7],P[8:11]), P[0])
h[12] = h[11] # to over write state 12 enthalpy with state 11 because isenthalpic expansion.
enthalpy_2 = np.append(h[11:15],h[2])
pressure_2 = np.append(P[11:15],P[2])

plt.plot(enthalpy_1,pressure_1,'-gX',linewidth=1.5,alpha=0.9,label='R-410A')    
plt.plot(enthalpy_2,pressure_2,'-gX',linewidth=1.5,alpha=0.9,label='R-410A')
   
#states numbers
plt.text(433, 750, '1',fontsize=10)
plt.text(482, 3700, '4',fontsize=10)
plt.text(268, 3700, '5,6,9',fontsize=10)
plt.text(251, 3550, '7',fontsize=10)
plt.text(263, 1500, '10',fontsize=10)
plt.text(251, 750, '8',fontsize=10)
plt.text(460, 1500, '2',fontsize=10)
plt.text(450, 1800, '3',fontsize=10)
plt.text(430, 1800, '11',fontsize=10)
   
   
plt.ylim(500,6500)
plt.xlim(200,500)
plt.yscale('log')
plt.xlabel(r'$h$ [kJ kg$^{-1}$]')
plt.ylabel(r'$P$ [kPa]')
plt.yticks([500, 1000, 2000, 5000],
           [r'$500$', r'$1000$', r'$2000$', r'$5000$'])
plt.text(470,6000,'R-410A',ha='center',va='top')
# leg=plt.legend(loc='upper left',fancybox=False,numpoints=1)
# frame=leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('p-h')    
plt.show()
plt.close()



# ######################
# #Plot T-s diagram
# ######################
# ts_plot_R410A= PropertyPlot(ref_fluid, 'Ts',unit_system='KSI')
# ts_plot_R410A.calc_isolines(CoolProp.iQ, num=2)
# ts_plot_R410A.draw() #actually draw isoline
# ts_plot_R410A.isolines.clear() #stop isoline, to avoid ploting the isoline at the end 
# ts_plot_R410A.title('T-s R410A')
# ts_plot_R410A.xlabel(r'$s$ [kJ kg$^{-1}$ K$^{-1}$]')#r'$s$ [{kJ} {kg$^{-1}$ K$^{-1}$}]'
# ts_plot_R410A.ylabel(r'$T$ [$\degree$C]')#{\textdegree}C
#ts_plot_R410A.grid()
plt.plot(sx0,Tx0,color="k",linewidth=1)
plt.plot(sx1,Tx1,color="k",linewidth=1)
plt.plot(sx8,Tx8,color="grey",linewidth=0.25)
plt.text(1.63, -5, '$x$=0.8',color="grey",fontsize=6,rotation=90)
plt.plot(sx6,Tx6,color="grey",linewidth=0.25)
plt.text(1.45, -5, '$x$=0.6',color="grey",fontsize=6,rotation=75)
plt.plot(sx4,Tx4,color="grey",linewidth=0.25)
plt.text(1.27, -5, '$x$=0.4',color="grey",fontsize=6,rotation=60)
plt.plot(sx2,Tx2,color="grey",linewidth=0.25)
plt.text(1.095, -5, '$x$=0.2',color="grey",fontsize=6,rotation=50)


s[9] = PropsSI('S','P',P[9]*1000,'H',h[9]*1000,ref_fluid)/1000
# T[9] = PropsSI('T','P',P[9]*1000,'S',s[9]*1000,ref_fluid)-273.15
entropy_1 = np.append(np.append(s[0:7],s[8:11]), s[0])
temp_1 = np.append(np.append(T[0:7],T[8:11]), T[0])
s[12] = PropsSI('S','P',P[12]*1000,'H',h[12]*1000,ref_fluid)/1000
# T[12] = PropsSI('T','P',P[12]*1000,'S',s[12]*1000,ref_fluid)-273.15
entropy_2 = np.append(s[11:15],s[2])
temp_2 = np.append(T[11:15],T[2])

plt.plot(entropy_1,temp_1,'-rP',linewidth=1.5,alpha=0.9,label='R-410A')    
plt.plot(entropy_2,temp_2,'-rP',linewidth=1.5,alpha=0.9,label='R-410A')

  
#states numbers
plt.text(1.85, 8, '1',fontsize=10)
plt.text(1.89, 94, '4',fontsize=10)
plt.text(1.17, 48, '5,6,9',fontsize=10)
plt.text(1.16, 37, '7',fontsize=10)
plt.text(1.21, 22, '10',fontsize=10)
plt.text(1.18, 0, '8',fontsize=10)
plt.text(1.76, 37, '11',fontsize=10)
plt.text(1.87, 51, '2',fontsize=10)
plt.text(1.82, 47, '3',fontsize=10)
  
plt.ylim([-20,100])
plt.xlim([1.0,2.0])
plt.xlabel(r'$s$ [kJ kg$^{-1}$ K$^{-1}$]')#r'$s$ [{kJ} {kg$^{-1}$ K$^{-1}$}]'
plt.ylabel(r'$T$ [$\degree$C]')#{\textdegree}C
plt.text(1.1,95,'R-410A',ha='center',va='top')
# leg=plt.legend(loc='upper left',fancybox=False, numpoints=1)
# frame=leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('T-s')    
plt.show()
'''
Created on March 7, 2017

@author: ammarbahman

Note: this file plots the Ts/Ph diagrams  in 60K ECU for IJR paper

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CoolProp
from CoolProp.Plots import PropertyPlot
import matplotlib as mpl
mpl.style.use('classic')

#===============================================================================
# Latex render
#===============================================================================
#mpl.use('pgf')

def figsize(scale):
    fig_width_pt = 469.755                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
"text.usetex": True,                # use LaTeX to write all text
"font.family": "serif",
"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
"font.sans-serif": [],
"font.monospace": [],
"axes.labelsize": 10,               # LaTeX default is 10pt font.
"font.size": 10,
"legend.fontsize": 8,               # Make the legend/label fonts a little smaller
"legend.labelspacing":0.2,
"xtick.labelsize": 8,
"ytick.labelsize": 8,
"figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
"pgf.preamble": [
r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)
#===============================================================================
# END of Latex render
#===============================================================================

df = pd.read_excel("Ts_Ph_test1.xlsx")
B_P = np.array(df[:12]["P"])
B_T = np.array(df[:12]["T"])
B_h = np.array(df[:12]["h"])
B_s = np.array(df[:12]["s"])

M_P = np.array(df[16:28]["P"])
M_T = np.array(df[16:28]["T"])
M_h = np.array(df[16:28]["h"])
M_s = np.array(df[16:28]["s"])

I_P = np.array(df[32:44]["P"])
I_T = np.array(df[32:44]["T"])
I_h = np.array(df[32:44]["h"])
I_s = np.array(df[32:44]["s"])

df = pd.read_excel("x=0.8.xlsx")
Px8 = np.array(df[:]["p"])
hx8 = np.array(df[:]["h"])
Tx8 = np.array(df[:]["T"])
sx8 = np.array(df[:]["s"])

df = pd.read_excel("x=0.6.xlsx")
Px6 = np.array(df[:]["p"])
hx6 = np.array(df[:]["h"])
Tx6 = np.array(df[:]["T"])
sx6 = np.array(df[:]["s"])

df = pd.read_excel("x=0.4.xlsx")
Px4 = np.array(df[:]["p"])
hx4 = np.array(df[:]["h"])
Tx4 = np.array(df[:]["T"])
sx4 = np.array(df[:]["s"])

df = pd.read_excel("x=0.2.xlsx")
Px2 = np.array(df[:]["p"])
hx2 = np.array(df[:]["h"])
Tx2 = np.array(df[:]["T"])
sx2 = np.array(df[:]["s"])

df = pd.read_excel("x=0.xlsx")
Tx0 = np.array(df[:]["T"])
sx0 = np.array(df[:]["s"])

df = pd.read_excel("x=1.xlsx")
Tx1 = np.array(df[:]["T"])
sx1 = np.array(df[:]["s"])
               
ref_fluid = 'HEOS::R407C'
#Plot P-h diagram 
ph_plot_R407C = PropertyPlot(ref_fluid, 'Ph')
ph_plot_R407C.calc_isolines(CoolProp.iQ, num=2)
ph_plot_R407C.title('P-h R407C')
ph_plot_R407C.xlabel(r'$h$ [{kJ}/{kg}]')
ph_plot_R407C.ylabel(r'$P$ [kPa]')
ph_plot_R407C.axis.set_yscale('log')
#ph_plot_R407C.grid()
plt.plot(B_h,B_P,'-',linewidth=1.5,label='Baseline')
plt.plot(M_h,M_P,'r--',linewidth=1.5, label='Modified')
plt.plot(I_h,I_P,'g:',linewidth=1.5,label='Interleaved')
plt.plot(hx8,Px8,color="grey",linewidth=0.25)
plt.text(364, 600, 'x=0.8',color="grey",fontsize=5,rotation=70)
plt.plot(hx6,Px6,color="grey",linewidth=0.25)
plt.text(322, 600, 'x=0.6',color="grey",fontsize=5,rotation=67)
plt.plot(hx4,Px4,color="grey",linewidth=0.25)
plt.text(279, 600, 'x=0.4',color="grey",fontsize=5,rotation=64)
plt.plot(hx2,Px2,color="grey",linewidth=0.25)
plt.ylim(500,5000)
plt.xlim(250,500)
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
ts_plot_R407C.title('T-s R407C')
ts_plot_R407C.xlabel(r'$s$ [{kJ}/{kg-K}]')
ts_plot_R407C.ylabel(r'$T$ [$^{\circ}$C]')
#ts_plot_R407C.grid()
plt.plot(B_s,B_T,'-',linewidth=1.5,label='Baseline')
plt.plot(M_s,M_T,'r--',linewidth=1.5, label='Modified')
plt.plot(I_s,I_T,'g:',linewidth=1.5,label='Interleaved')
plt.plot(sx0,Tx0,color="k",linewidth=0.6)
plt.plot(sx1,Tx1,color="k",linewidth=0.6)
plt.plot(sx8,Tx8,color="grey",linewidth=0.25)
plt.text(1.608, -5, 'x=0.8',color="grey",fontsize=5,rotation=90)
plt.plot(sx6,Tx6,color="grey",linewidth=0.25)
plt.text(1.431, -5, 'x=0.6',color="grey",fontsize=5,rotation=65)
plt.plot(sx4,Tx4,color="grey",linewidth=0.25)
plt.text(1.26, -5, 'x=0.4',color="grey",fontsize=5,rotation=48)
plt.plot(sx2,Tx2,color="grey",linewidth=0.25)

plt.ylim([-20,120])
plt.xlim([1.2,2.0])
plt.title('')
plt.text(1.90,-5,'R-407C',ha='center',va='top')
leg=plt.legend(loc='upper left',fancybox=False, numpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout() 
ts_plot_R407C.savefig('T-s.pdf')    
ts_plot_R407C.show()


    
    
##########plot superheat -- Baseline##########
# plt.plot(np.arange(1,7,1),B_testC,'--ko',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test C')
# plt.errorbar(np.arange(1,7,1),B_testC,yerr=0.5,fmt='',linestyle="None",color='k')
# plt.plot(np.arange(1,7,1),B_testB,'--bs',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test B')
# plt.errorbar(np.arange(1,7,1),B_testB,yerr=0.5,fmt='',linestyle="None",color='b')
# plt.plot(np.arange(1,7,1),B_test6,'--r^',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 6')
# plt.errorbar(np.arange(1,7,1),B_test6,yerr=0.5,fmt='',linestyle="None",color='r')
# plt.plot(np.arange(1,7,1),B_test5,'--g*',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 5')
# plt.errorbar(np.arange(1,7,1),B_test5,yerr=0.5,fmt='',linestyle="None",color='g')
# plt.plot(np.arange(1,7,1),B_test4,'--P',markersize=5,markeredgewidth=0.1,alpha=0.9,color='brown',label=r'Test 4')
# plt.errorbar(np.arange(1,7,1),B_test4,yerr=0.5,fmt='',linestyle="None",color='brown')
# plt.plot(np.arange(1,7,1),B_test3,'--cH',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 3')
# plt.errorbar(np.arange(1,7,1),B_test3,yerr=0.5,fmt='',linestyle="None",color='c')
# plt.plot(np.arange(1,7,1),B_test2,'--yD',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 2')
# plt.errorbar(np.arange(1,7,1),B_test2,yerr=0.5,fmt='',linestyle="None",color='y')
# plt.plot(np.arange(1,7,1),B_test1,'--mX',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 1')
# plt.errorbar(np.arange(1,7,1),B_test1,yerr=0.5,fmt='',linestyle="None",color='m')
# plt.ylim(0,25)
# plt.xlim(0,7)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7],
#            [r'$top$', r'$1$', r'$2$', r'$3$',r'$4$', r'$5$', r'$6$', r'$bottom$'])
# plt.xlabel('Circuit number')
# plt.ylabel('$T_{sup}$ [$^{\circ}$C]')
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.savefig('T_sup_baseline.pdf')
# plt.show()



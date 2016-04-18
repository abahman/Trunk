'''
Created on Apr 17, 2015

@author: ammarbahman

Note: you need to have all the test results CSV files executed before run this file.

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import csv2rec


#===============================================================================
# Latex render
#===============================================================================
import matplotlib as mpl
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

#Experimental Results
T_sup = np.array([5,10,15])
#Import data from CSV file
data1 = csv2rec('60K_Ammar_purdue_conf_linear_T_sup=5.0.csv',delimiter=',')
data2 = csv2rec('60K_Ammar_purdue_conf_linear_T_sup=10.0.csv',delimiter=',')
data3 = csv2rec('60K_Ammar_purdue_conf_linear_T_sup=15.0.csv',delimiter=',')
#Arrange data in Numpy array for the 5 different tests
capacity_uniform = np.array([data1[2][6],data2[2][6],data3[2][6]])
capacity_baseline = np.array([data1[3][6],data2[3][6],data3[3][6]])
capacity_interleave = np.array([data1[4][6],data2[4][6],data3[4][6]])
#to convert string array to integer array
capacity_uniform = capacity_uniform.astype(np.float)
capacity_baseline = capacity_baseline.astype(np.float)
capacity_interleave = capacity_interleave.astype(np.float)
#plots
#Plot cooling capacity comparison
plt.plot(T_sup,capacity_uniform,'-ob',label='Uniform')
plt.plot(T_sup,capacity_baseline,'--or',label='Baseline')
plt.plot(T_sup,capacity_interleave,'-ok',label='Interleaved')
#plt.ylim(0.025,0.055)
plt.xlim(0,20)
plt.legend(loc='best',fancybox=False)
plt.xlabel(r'$T_{sup}$ [\textdegree$\mathrm{C}]$')
plt.ylabel(r'$\dot Q$ $[\mathrm{W}]$')
plt.title(r'Effect of changing $T_{sup}$ for linear profile $T_{H}$=125[\textdegree$\mathrm{F}]$ $T_{L}$=90[\textdegree$\mathrm{F}]$ $T_{sup}$=5-15[\textdegree$\mathrm{C}]$')
plt.savefig('plots/purdue_conf_1_linear.pdf')
plt.show()

#Experimental Results
T_sup = np.array([5,10,15])
#Import data from CSV file
data1 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_sup=5.csv',delimiter=',')
data2 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_sup=10.csv',delimiter=',')
data3 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_sup=15.csv',delimiter=',')
#Arrange data in Numpy array for the 5 different tests
capacity_uniform = np.array([data1[2][6],data2[2][6],data3[2][6]])
capacity_baseline = np.array([data1[3][6],data2[3][6],data3[3][6]])
capacity_interleave = np.array([data1[4][6],data2[4][6],data3[4][6]])
#to convert string array to integer array
capacity_uniform = capacity_uniform.astype(np.float)
capacity_baseline = capacity_baseline.astype(np.float)
capacity_interleave = capacity_interleave.astype(np.float)
#plots
#Plot cooling capacity comparison
plt.plot(T_sup,capacity_uniform,'-ob',label='Uniform')
plt.plot(T_sup,capacity_baseline,'--or',label='Baseline')
plt.plot(T_sup,capacity_interleave,'-ok',label='Interleaved')
#plt.ylim(0.025,0.055)
plt.xlim(0,20)
plt.legend(loc='best',fancybox=False)
plt.xlabel(r'$T_{sup}$ [\textdegree$\mathrm{C}]$')
plt.ylabel(r'$\dot Q$ $[\mathrm{W}]$')
plt.title(r'Effect of changing $T_{sup}$ for measured profile $T_{H}$=125 [\textdegree$\mathrm{F}]$ $T_{L}$=90 [\textdegree$\mathrm{F}]$ $T_{sup}$=5-15 [\textdegree$\mathrm{C}]$')
plt.savefig('plots/purdue_conf_1.pdf')
plt.show()

#Experimental Results
T_env = np.array([75,85,95,115,125])
#Import data from CSV file
data1 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=fixed1_T_out=75.0_T_sup=10.csv',delimiter=',')
data2 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=fixed1_T_out=85.0_T_sup=10.csv',delimiter=',')
data3 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=fixed1_T_out=95.0_T_sup=10.csv',delimiter=',')
data4 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=fixed1_T_out=115.0_T_sup=10.csv',delimiter=',')
data5 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=fixed1_T_out=125.0_T_sup=10.csv',delimiter=',')
#Arrange data in Numpy array for the 5 different tests
capacity_uniform = np.array([data1[2][6],data2[2][6],data3[2][6],data4[2][6],data5[2][6]])
capacity_baseline = np.array([data1[3][6],data2[3][6],data3[3][6],data4[3][6],data5[3][6]])
capacity_interleave = np.array([data1[4][6],data2[4][6],data3[4][6],data4[4][6],data5[4][6]])
#to convert string array to integer array
capacity_uniform = capacity_uniform.astype(np.float)
capacity_baseline = capacity_baseline.astype(np.float)
capacity_interleave = capacity_interleave.astype(np.float)
#plots
#Plot cooling capacity comparison
plt.plot(T_env,capacity_uniform,'-ob',label='Uniform')
plt.plot(T_env,capacity_baseline,'--or',label='Baseline')
plt.plot(T_env,capacity_interleave,'-ok',label='Interleaved')
#plt.ylim(0.025,0.055)
plt.xlim(70,130)
plt.legend(loc='best',fancybox=False)
plt.xlabel(r'$T_{H}$ [\textdegree$\mathrm{F}]$')
plt.ylabel(r'$\dot Q$ $[\mathrm{W}]$')
plt.title(r'$T_{sup}$=10 [\textdegree$\mathrm{C}]$ $T_L$=80 [\textdegree$\mathrm{F}]$ (RH=51.07$\%$)-wetcoil $T_H$=75-125 [\textdegree$\mathrm{F}]$')
plt.savefig('plots/purdue_conf_2.pdf')
plt.show()

#Experimental Results
T_env = np.array([75,85,95,115,125])
#Import data from CSV file
data1 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=fixed2_T_out=75.0_T_sup=10.csv',delimiter=',')
data2 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=fixed2_T_out=85.0_T_sup=10.csv',delimiter=',')
data3 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=fixed2_T_out=95.0_T_sup=10.csv',delimiter=',')
data4 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=fixed2_T_out=115.0_T_sup=10.csv',delimiter=',')
data5 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=fixed2_T_out=125.0_T_sup=10.csv',delimiter=',')
#Arrange data in Numpy array for the 5 different tests
capacity_uniform = np.array([data1[2][6],data2[2][6],data3[2][6],data4[2][6],data5[2][6]])
capacity_baseline = np.array([data1[3][6],data2[3][6],data3[3][6],data4[3][6],data5[3][6]])
capacity_interleave = np.array([data1[4][6],data2[4][6],data3[4][6],data4[4][6],data5[4][6]])
#to convert string array to integer array
capacity_uniform = capacity_uniform.astype(np.float)
capacity_baseline = capacity_baseline.astype(np.float)
capacity_interleave = capacity_interleave.astype(np.float)
#plots
#Plot cooling capacity comparison
plt.plot(T_env,capacity_uniform,'-ob',label='Uniform')
plt.plot(T_env,capacity_baseline,'--or',label='Baseline')
plt.plot(T_env,capacity_interleave,'-ok',label='Interleaved')
#plt.ylim(0.025,0.055)
plt.xlim(70,130)
plt.legend(loc='best',fancybox=False)
plt.xlabel(r'$T_{H}$ [\textdegree$\mathrm{F}]$')
plt.ylabel(r'$\dot Q$ $[\mathrm{W}]$')
plt.title(r'$T_{sup}$=10 [\textdegree$\mathrm{C}]$ $T_L$=80 [\textdegree$\mathrm{F}]$(RH=21.51$\%$)-drycoil $T_H$=75-125 [$\textdegree\mathrm{F}]$')
plt.savefig('plots/purdue_conf_3.pdf')
plt.show()

#Experimental Results
T_in = np.array([75,80,85,90])
#Import data from CSV file
data1 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=vary1_T_out=125.0a_T_sup=10.csv',delimiter=',')
data2 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=vary1_T_out=125.0b_T_sup=10.csv',delimiter=',')
data3 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=vary1_T_out=125.0c_T_sup=10.csv',delimiter=',')
data4 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=vary1_T_out=125.0d_T_sup=10.csv',delimiter=',')
#Arrange data in Numpy array for the 5 different tests
capacity_uniform = np.array([data1[2][6],data2[2][6],data3[2][6],data4[2][6]])
capacity_baseline = np.array([data1[3][6],data2[3][6],data3[3][6],data4[3][6]])
capacity_interleave = np.array([data1[4][6],data2[4][6],data3[4][6],data4[4][6]])
#to convert string array to integer array
capacity_uniform = capacity_uniform.astype(np.float)
capacity_baseline = capacity_baseline.astype(np.float)
capacity_interleave = capacity_interleave.astype(np.float)
#plots
#Plot cooling capacity comparison
plt.plot(T_in,capacity_uniform,'-ob',label='Uniform')
plt.plot(T_in,capacity_baseline,'--or',label='Baseline')
plt.plot(T_in,capacity_interleave,'-ok',label='Interleaved')
plt.ylim(10000,11800)
plt.xlim(70,95)
plt.legend(loc='best',fancybox=False)
plt.xlabel(r'$T_{L}$ [\textdegree$\mathrm{F}]$')
plt.ylabel(r'$\dot Q$ $[\mathrm{W}]$')
plt.title(r'$T_{sup}$=10 [\textdegree$\mathrm{C}]$, $T_H$=125 [\textdegree$\mathrm{F}]$ $T_L$=75-90 [\textdegree$\mathrm{F}]$(RH=21.51$\%$)-drycoil')
plt.savefig('plots/purdue_conf_4.pdf')
plt.show()

#Experimental Results
T_in = np.array([75,80,85,90])
#Import data from CSV file
data1 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=vary2_T_out=125.0a_T_sup=10.csv',delimiter=',')
data2 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=vary2_T_out=125.0b_T_sup=10.csv',delimiter=',')
data3 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=vary2_T_out=125.0c_T_sup=10.csv',delimiter=',')
data4 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_T_in=vary2_T_out=125.0d_T_sup=10.csv',delimiter=',')
#Arrange data in Numpy array for the 5 different tests
capacity_uniform = np.array([data1[2][6],data2[2][6],data3[2][6],data4[2][6]])
capacity_baseline = np.array([data1[3][6],data2[3][6],data3[3][6],data4[3][6]])
capacity_interleave = np.array([data1[4][6],data2[4][6],data3[4][6],data4[4][6]])
#to convert string array to integer array
capacity_uniform = capacity_uniform.astype(np.float)
capacity_baseline = capacity_baseline.astype(np.float)
capacity_interleave = capacity_interleave.astype(np.float)
#plots
#Plot cooling capacity comparison
plt.plot(T_in,capacity_uniform,'-ob',label='Uniform')
plt.plot(T_in,capacity_baseline,'--or',label='Baseline')
plt.plot(T_in,capacity_interleave,'-ok',label='Interleaved')
plt.ylim(15000,16600)
plt.xlim(70,95)
plt.legend(loc='best',fancybox=False)
plt.xlabel(r'$T_{L}$ [\textdegree$\mathrm{F}]$')
plt.ylabel(r'$\dot Q$ $[\mathrm{W}]$')
plt.title(r'$T_{sup}$=10 [\textdegree$\mathrm{C}]$ $T_H$=125 [\textdegree$\mathrm{F}]$ $T_L$=75-90 [\textdegree$\mathrm{F}]$ (RH=51.07$\%$)-wetcoil')
plt.savefig('plots/purdue_conf_5.pdf')
plt.show()

#Experimental Results
RH_in = np.array([0.01,25,50,75,99.9])
#Import data from CSV file
data1 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_RH6a_T_out=125.0_T_sup=10.csv',delimiter=',')
data2 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_RH6b_T_out=125.0_T_sup=10.csv',delimiter=',')
data3 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_RH6c_T_out=125.0_T_sup=10.csv',delimiter=',')
data4 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_RH6d_T_out=125.0_T_sup=10.csv',delimiter=',')
data5 = csv2rec('60K-6Circuit_airMD_Ammar_purdue_conf_RH6e_T_out=125.0_T_sup=10.csv',delimiter=',')
#Arrange data in Numpy array for the 5 different tests
capacity_uniform = np.array([data1[2][6],data2[2][6],data3[2][6],data4[2][6],data5[2][6]])
capacity_baseline = np.array([data1[3][6],data2[3][6],data3[3][6],data4[3][6],data5[2][6]])
capacity_interleave = np.array([data1[4][6],data2[4][6],data3[4][6],data4[4][6],data5[2][6]])
#to convert string array to integer array
capacity_uniform = capacity_uniform.astype(np.float)
capacity_baseline = capacity_baseline.astype(np.float)
capacity_interleave = capacity_interleave.astype(np.float)
#plots
#Plot cooling capacity comparison
plt.plot(RH_in,capacity_uniform,'-ob',label='Uniform')
plt.plot(RH_in,capacity_baseline,'--or',label='Baseline')
plt.plot(RH_in,capacity_interleave,'-ok',label='Interleaved')
#plt.plot(np.array([47.64]),np.array([18050.0]),'Dy',label='Baseline (experimental)')
plt.ylim(5000,45000)
plt.xlim(0,100)
plt.legend(loc='best',fancybox=False)
plt.xlabel(r'$\mathrm{RH}_{L}$ $[\%]$')
plt.ylabel(r'$\dot Q$ $[\mathrm{W}]$')
plt.title(r'Effect of changing humidity $T_{sup}$=10 [\textdegree$\mathrm{C}]$ $T_H$=125[\textdegree$\mathrm{F}]$ $T_L$=90 [\textdegree$\mathrm{F}]$')
plt.savefig('plots/purdue_conf_6.pdf')
plt.show()

# #Combine
# fig = plt.figure(1, figsize=(10, 10), dpi=100)
# for i, gtype in enumerate(['Mass', 'Capacity', 'Power', 'Compressor', 'COPS','Charge']):
#     ax = plt.subplot(3, 2, i+1)
#     if gtype.startswith('Mass'):
#         plt.plot(T_env,m_dot_exp,'-ob',label='Experimental')
#         plt.errorbar(T_env,m_dot_exp, yerr=0.002*m_dot_exp)
#         plt.plot(T_env,m_dot,'--or',label='Model')
#         plt.ylim(0.02,0.05)
#         plt.xlim(70,130)
#         plt.legend(loc='best',fancybox=False)
#         plt.xlabel(r'$T_{env}$ $[\textdegree\mathrm{F}]$')
#         plt.ylabel(r'$\dot m_r$ $[\mathrm{kg/s}]$')
#         #plt.title('Mass flowrate Comparison')
#     if gtype.startswith('Capacity'):
#         plt.plot(T_env,capacity_exp,'-ob',label='Experimental')
#         plt.errorbar(T_env,capacity_exp, yerr=0.00905522*capacity_exp)
#         plt.plot(T_env,capacity,'--or',label='Model')
#         plt.ylim(4000,7000)
#         plt.xlim(70,130)
#         plt.legend(loc='best',fancybox=False)
#         plt.xlabel(r'$T_{env}$ $[\textdegree\mathrm{F}]$')
#         plt.ylabel(r'$\dot Q_{evap}$ $[\mathrm{W}]$')
#         #plt.title('Capacity Comparison')
#     if gtype.startswith('Power'):
#         plt.plot(T_env,total_power_exp,'-ob',label='Experimental')
#         plt.errorbar(T_env,total_power_exp, yerr=0.02618715*total_power_exp)
#         plt.plot(T_env,total_power,'--or',label='Model')
#         plt.ylim(2000,5000)
#         plt.xlim(70,130)
#         plt.legend(loc='best',fancybox=False)
#         plt.xlabel(r'$T_{env}$ $[\textdegree\mathrm{F}]$')
#         plt.ylabel(r'$\dot E_t$ $[\mathrm{W}]$')
#         #plt.title('Total Power Comparison')
#     if gtype.startswith('Compressor'):
#         plt.plot(T_env,compressor_power_exp,'-ob',label='Experimental')
#         plt.errorbar(T_env,compressor_power_exp, yerr=112.5)
#         plt.plot(T_env,compressor_power,'--or',label='Model')
#         plt.ylim(1000,3500)
#         plt.xlim(70,130)
#         plt.legend(loc='best',fancybox=False)
#         plt.xlabel(r'$T_{env}$ $[\textdegree\mathrm{F}]$')
#         plt.ylabel(r'$\dot W_{comp}$ $[\mathrm{W}]$')
#         #plt.title('Compressor Power Comparison')
#     if gtype.startswith('COPS'):
#         plt.plot(T_env,COPS_exp,'-ob',label='Experimental')
#         plt.errorbar(T_env,COPS_exp, yerr=0.02772727*COPS_exp)
#         plt.plot(T_env,COPS,'--or',label='Model')
#         plt.ylim(1,2.4)
#         plt.xlim(70,130)
#         plt.legend(loc='best',fancybox=False)
#         plt.xlabel(r'$T_{env}$ $[\textdegree\mathrm{F}]$')
#         plt.ylabel(r'$\mathrm{COP}_{sys}$')
#         #plt.title('System COP Comparison')
#     if gtype.startswith('Charge'):
#         plt.plot(T_env,charge_exp,'-ob',label='Experimental')
#         plt.errorbar(T_env,charge_exp, yerr=0.0)
#         plt.plot(T_env,charge,'--or',label='Model')
#         plt.ylim(0,1.6)
#         plt.xlim(70,130)
#         plt.legend(loc='best',fancybox=False)
#         plt.xlabel(r'$T_{env}$ $[\textdegree\mathrm{F}]$')
#         plt.ylabel(r'$\mathrm{Charge}$ $[\mathrm{kg}]$')
#         #plt.title('System charge Comparison')
# fig.set_tight_layout(True)
# plt.savefig('images/comined_comparison.pdf')
# plt.show()
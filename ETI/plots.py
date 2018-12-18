import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import exp
import scipy
import matplotlib as mpl
mpl.style.use('classic')
mpl.style.use('Elsevier.mplstyle')
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['figure.figsize'] = [6,4]

# #===============================================================================
# # Latex render
# #===============================================================================
# import matplotlib as mpl
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
 

#import the excel file
df = pd.read_excel("data_2ph_injection.xlsx",sheet_name='All_data')

T_dis_test1 = np.array(df[1:5]["T_comp_out"])
T_dis_test2 = np.array(df[5:9]["T_comp_out"])
T_dis_test3 = np.array(df[9:13]["T_comp_out"])
T_dis_test4 = np.array(df[13:17]["T_comp_out"])
T_dis_test5 = np.array(df[17:21]["T_comp_out"])
T_dis_test6 = np.array(df[21:25]["T_comp_out"])
T_dis_testB = np.array(df[25:29]["T_comp_out"])
T_dis_testC = np.array(df[29:33]["T_comp_out"])

Q_dot_evap_test1 = np.array(df[1:5]["Q_dot_evap"])
Q_dot_evap_test2 = np.array(df[5:9]["Q_dot_evap"])
Q_dot_evap_test3 = np.array(df[9:13]["Q_dot_evap"])
Q_dot_evap_test4 = np.array(df[13:17]["Q_dot_evap"])
Q_dot_evap_test5 = np.array(df[17:21]["Q_dot_evap"])
Q_dot_evap_test6 = np.array(df[21:25]["Q_dot_evap"])
Q_dot_evap_testB = np.array(df[25:29]["Q_dot_evap"])
Q_dot_evap_testC = np.array(df[29:33]["Q_dot_evap"])

W_dot_comp_test1 = np.array(df[1:5]["W_dot_comp"])
W_dot_comp_test2 = np.array(df[5:9]["W_dot_comp"])
W_dot_comp_test3 = np.array(df[9:13]["W_dot_comp"])
W_dot_comp_test4 = np.array(df[13:17]["W_dot_comp"])
W_dot_comp_test5 = np.array(df[17:21]["W_dot_comp"])
W_dot_comp_test6 = np.array(df[21:25]["W_dot_comp"])
W_dot_comp_testB = np.array(df[25:29]["W_dot_comp"])
W_dot_comp_testC = np.array(df[29:33]["W_dot_comp"])

COP_sys_r_test1 = np.array(df[1:5]["COP_sys_r"])
COP_sys_r_test2 = np.array(df[5:9]["COP_sys_r"])
COP_sys_r_test3 = np.array(df[9:13]["COP_sys_r"])
COP_sys_r_test4 = np.array(df[13:17]["COP_sys_r"])
COP_sys_r_test5 = np.array(df[17:21]["COP_sys_r"])
COP_sys_r_test6 = np.array(df[21:25]["COP_sys_r"])
COP_sys_r_testB = np.array(df[25:29]["COP_sys_r"])
COP_sys_r_testC = np.array(df[29:33]["COP_sys_r"])

##########plot discharge temperature##########
plt.plot(np.arange(1,5,1),T_dis_testC,'--ko',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test C')
plt.errorbar(np.arange(1,5,1),T_dis_testC,yerr=1.1,fmt='',linestyle="None",color='k')
plt.plot(np.arange(1,5,1),T_dis_testB,'--bs',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test B')
plt.errorbar(np.arange(1,5,1),T_dis_testB,yerr=1.1,fmt='',linestyle="None",color='b')
plt.plot(np.arange(1,5,1),T_dis_test6,'--r^',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 6')
plt.errorbar(np.arange(1,5,1),T_dis_test6,yerr=1.1,fmt='',linestyle="None",color='r')
plt.plot(np.arange(1,5,1),T_dis_test5,'--g*',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 5')
plt.errorbar(np.arange(1,5,1),T_dis_test5,yerr=1.1,fmt='',linestyle="None",color='g')
plt.plot(np.arange(1,5,1),T_dis_test4,'--P',markersize=5,markeredgewidth=0.1,alpha=0.9,color='brown',label=r'Test 4')
plt.errorbar(np.arange(1,5,1),T_dis_test4,yerr=1.1,fmt='',linestyle="None",color='brown')
plt.plot(np.arange(1,5,1),T_dis_test3,'--cH',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 3')
plt.errorbar(np.arange(1,5,1),T_dis_test3,yerr=1.1,fmt='',linestyle="None",color='c')
plt.plot(np.arange(1,5,1),T_dis_test2,'--yD',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 2')
plt.errorbar(np.arange(1,5,1),T_dis_test2,yerr=1.1,fmt='',linestyle="None",color='y')
plt.plot(np.arange(1,5,1),T_dis_test1,'--mX',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 1')
plt.errorbar(np.arange(1,5,1),T_dis_test1,yerr=1.1,fmt='',linestyle="None",color='m')
plt.ylim(60,120)
plt.xlim(0,6)
plt.xticks([0, 1, 2, 3, 4, 5],
           ['', 'No\n Injection', 'Saturated\n Injection', '2-Phase\n Injection\n (88%)','2-phase\n Injection\n (78%)', ''])
plt.xlabel('')
plt.ylabel('$T_{dis}$ [$\degree$C]')
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()
plt.savefig('T_dis.pdf')
plt.show()


##########plot cooling capacity##########
plt.plot(np.arange(1,5,1),Q_dot_evap_testC,'--ko',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test C')
plt.errorbar(np.arange(1,5,1),Q_dot_evap_testC,yerr=0.0091*Q_dot_evap_testC,fmt='',linestyle="None",color='k')
plt.plot(np.arange(1,5,1),Q_dot_evap_testB,'--bs',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test B')
plt.errorbar(np.arange(1,5,1),Q_dot_evap_testB,yerr=0.0091*Q_dot_evap_testB,fmt='',linestyle="None",color='b')
plt.plot(np.arange(1,5,1),Q_dot_evap_test6,'--r^',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 6')
plt.errorbar(np.arange(1,5,1),Q_dot_evap_test6,yerr=0.0091*Q_dot_evap_test6,fmt='',linestyle="None",color='r')
plt.plot(np.arange(1,5,1),Q_dot_evap_test5,'--g*',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 5')
plt.errorbar(np.arange(1,5,1),Q_dot_evap_test5,yerr=0.0091*Q_dot_evap_test5,fmt='',linestyle="None",color='g')
plt.plot(np.arange(1,5,1),Q_dot_evap_test4,'--P',markersize=5,markeredgewidth=0.1,alpha=0.9,color='brown',label=r'Test 4')
plt.errorbar(np.arange(1,5,1),Q_dot_evap_test4,yerr=0.0091*Q_dot_evap_test4,fmt='',linestyle="None",color='brown')
plt.plot(np.arange(1,5,1),Q_dot_evap_test3,'--cH',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 3')
plt.errorbar(np.arange(1,5,1),Q_dot_evap_test3,yerr=0.0091*Q_dot_evap_test3,fmt='',linestyle="None",color='c')
plt.plot(np.arange(1,5,1),Q_dot_evap_test2,'--yD',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 2')
plt.errorbar(np.arange(1,5,1),Q_dot_evap_test2,yerr=0.0091*Q_dot_evap_test2,fmt='',linestyle="None",color='y')
plt.plot(np.arange(1,5,1),Q_dot_evap_test1,'--mX',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 1')
plt.errorbar(np.arange(1,5,1),Q_dot_evap_test1,yerr=0.0091*Q_dot_evap_test1,fmt='',linestyle="None",color='m')
plt.ylim(12,20)
plt.xlim(0,6)
plt.xticks([0, 1, 2, 3, 4, 5],
           ['', 'No\n Injection', 'Saturated\n Injection', '2-Phase\n Injection\n (88%)','2-phase\n Injection\n (78%)', ''])
plt.xlabel('')
plt.ylabel('$\dot Q_{evap}$ [kW]')
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()
plt.savefig('Q_evap.pdf')
plt.show()

##########plot compressor work##########
plt.plot(np.arange(1,5,1),W_dot_comp_testC,'--ko',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test C')
plt.errorbar(np.arange(1,5,1),W_dot_comp_testC,yerr=0.1125,fmt='',linestyle="None",color='k')
plt.plot(np.arange(1,5,1),W_dot_comp_testB,'--bs',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test B')
plt.errorbar(np.arange(1,5,1),W_dot_comp_testB,yerr=0.1125,fmt='',linestyle="None",color='b')
plt.plot(np.arange(1,5,1),W_dot_comp_test6,'--r^',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 6')
plt.errorbar(np.arange(1,5,1),W_dot_comp_test6,yerr=0.1125,fmt='',linestyle="None",color='r')
plt.plot(np.arange(1,5,1),W_dot_comp_test5,'--g*',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 5')
plt.errorbar(np.arange(1,5,1),W_dot_comp_test5,yerr=0.1125,fmt='',linestyle="None",color='g')
plt.plot(np.arange(1,5,1),W_dot_comp_test4,'--P',markersize=5,markeredgewidth=0.1,alpha=0.9,color='brown',label=r'Test 4')
plt.errorbar(np.arange(1,5,1),W_dot_comp_test4,yerr=0.1125,fmt='',linestyle="None",color='brown')
plt.plot(np.arange(1,5,1),W_dot_comp_test3,'--cH',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 3')
plt.errorbar(np.arange(1,5,1),W_dot_comp_test3,yerr=0.1125,fmt='',linestyle="None",color='c')
plt.plot(np.arange(1,5,1),W_dot_comp_test2,'--yD',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 2')
plt.errorbar(np.arange(1,5,1),W_dot_comp_test2,yerr=0.1125,fmt='',linestyle="None",color='y')
plt.plot(np.arange(1,5,1),W_dot_comp_test1,'--mX',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 1')
plt.errorbar(np.arange(1,5,1),W_dot_comp_test1,yerr=0.1125,fmt='',linestyle="None",color='m')
plt.ylim(2,10)
plt.xlim(0,6)
plt.xticks([0, 1, 2, 3, 4, 5],
           ['', 'No\n Injection', 'Saturated\n Injection', '2-Phase\n Injection\n (88%)','2-phase\n Injection\n (78%)', ''])
plt.xlabel('')
plt.ylabel('$\dot W_{comp}$ [kW]')
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()
plt.savefig('W_comp.pdf')
plt.show()

##########plot system COP##########
plt.plot(np.arange(1,5,1),COP_sys_r_testC,'--ko',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test C')
plt.errorbar(np.arange(1,5,1),COP_sys_r_testC,yerr=0.0277*COP_sys_r_testC,fmt='',linestyle="None",color='k')
plt.plot(np.arange(1,5,1),COP_sys_r_testB,'--bs',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test B')
plt.errorbar(np.arange(1,5,1),COP_sys_r_testB,yerr=0.0277*COP_sys_r_testB,fmt='',linestyle="None",color='b')
plt.plot(np.arange(1,5,1),COP_sys_r_test6,'--r^',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 6')
plt.errorbar(np.arange(1,5,1),COP_sys_r_test6,yerr=0.0277*COP_sys_r_test6,fmt='',linestyle="None",color='r')
plt.plot(np.arange(1,5,1),COP_sys_r_test5,'--g*',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 5')
plt.errorbar(np.arange(1,5,1),COP_sys_r_test5,yerr=0.0277*COP_sys_r_test5,fmt='',linestyle="None",color='g')
plt.plot(np.arange(1,5,1),COP_sys_r_test4,'--P',markersize=5,markeredgewidth=0.1,alpha=0.9,color='brown',label=r'Test 4')
plt.errorbar(np.arange(1,5,1),COP_sys_r_test4,yerr=0.0277*COP_sys_r_test4,fmt='',linestyle="None",color='brown')
plt.plot(np.arange(1,5,1),COP_sys_r_test3,'--cH',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 3')
plt.errorbar(np.arange(1,5,1),COP_sys_r_test3,yerr=0.0277*COP_sys_r_test3,fmt='',linestyle="None",color='c')
plt.plot(np.arange(1,5,1),COP_sys_r_test2,'--yD',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 2')
plt.errorbar(np.arange(1,5,1),COP_sys_r_test2,yerr=0.0277*COP_sys_r_test2,fmt='',linestyle="None",color='y')
plt.plot(np.arange(1,5,1),COP_sys_r_test1,'--mX',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'Test 1')
plt.errorbar(np.arange(1,5,1),COP_sys_r_test1,yerr=0.0277*COP_sys_r_test1,fmt='',linestyle="None",color='m')
plt.ylim(1,4)
plt.xlim(0,6)
plt.xticks([0, 1, 2, 3, 4, 5],
           ['', 'No\n Injection', 'Saturated\n Injection', '2-Phase\n Injection\n (88%)','2-phase\n Injection\n (78%)', ''])
plt.xlabel('')
plt.ylabel('COP$_{sys}$ [-]')
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()
plt.savefig('COP_sys.pdf')
plt.show()
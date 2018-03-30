'''
Created on Jan 19, 2018

@author: ammarbahman


'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import csv2rec
import matplotlib as mpl
mpl.style.use('classic')
mpl.style.use('Elsevier.mplstyle')
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['figure.figsize'] = [6,4]

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

def rmse(predictions, targets):
    '''
    Root Mean Square Error
    '''
    n = len(predictions)
    RMSE = np.linalg.norm(predictions - targets) / np.sqrt(n) / np.mean(targets) * 100
    return RMSE

def mape(y_pred, y_true):  #maps==mean_absolute_percentage_error
    '''
    Mean Absolute Percentage Error
    '''
    MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return MAPE

#Experimental Results (superheated)
TestNo = np.arange(1,9,1)
m_dot_exp = np.array([0.1049,0.08967,0.08974,0.09389,0.08051,0.08206,0.09311,0.08494]) #[kg/s]
m_dot_tot_exp = np.array([0.1314,0.1134,0.1104,0.1106,0.09419,0.09385,0.1051,0.09789]) #[kg/s]
m_dot_inj_exp = m_dot_tot_exp - m_dot_exp #[kg/s]
cooling_capacity_exp = np.array([17.72,16.06,16.5,17.51,15.69,16.09,17.92,16.48]) #[kW]
total_power_exp = np.array([9.097,8.042,7.374,6.795,6.076,5.391,6.125,6.033]) #[kW]
compressor_power_exp = np.array([7.353,6.286,5.544,4.969,4.205,4.16,4.274,4.129]) #[kW]
COPS_exp = np.array([1.948,1.997,2.237,2.576,2.581,2.985,2.927,2.732]) #[-]
charge_exp = np.array([5.01,5.01,5.01,5.01,5.01,5.01,5.01,5.01]) #[kg]
heating_capacity_exp = np.array([25.08,22.38,22.08,22.5,19.85,20.2,22.12,20.54]) #[kW]
PHX_capacity_exp = np.array([4.757,4.49,3.975,3.255,2.763,2.391,2.383,2.602]) #[kW]
T_dis_exp = np.array([105.8,99.55,91.66,83.49,77.62,77.63,75.56,74.65]) #[C]

#Experimental Results (saturated)
TestNo = np.arange(1,9,1)
m_dot_exp_sat = np.array([104.5,89.57,89.56,93.34,80.26,80.67,93.48,83.15])/1000 #[kg/s]
m_dot_tot_exp_sat = np.array([137.3,117.4,113.0,112.2,95.23,93.36,107.4,97.05])/1000 #[kg/s]
m_dot_inj_exp_sat = m_dot_tot_exp_sat - m_dot_exp_sat #[kg/s]
cooling_capacity_exp_sat = np.array([17.57,15.96,16.38,17.38,15.63,15.96,17.84,16.27]) #[kW]
total_power_exp_sat = np.array([9.253,8.121,7.398,6.762,6.104,5.374,6.159,5.997]) #[kW]
compressor_power_exp_sat = np.array([7.517,6.316,5.583,4.958,4.208,4.136,4.291,4.108]) #[kW]
COPS_exp_sat = np.array([1.899,1.965,2.214,2.57,2.56,2.97,2.897,2.714]) #[-]
charge_exp_sat = np.array([5.01,5.01,5.01,5.01,5.01,5.01,5.01,5.01]) #[kg]
heating_capacity_exp_sat = np.array([25.09,22.33,22.02,22.39,19.82,20.05,22.09,20.34]) #[kW]
PHX_capacity_exp_sat = np.array([5.189,4.6,3.978,3.197,2.677,2.306,2.396,2.446]) #[kW]
T_dis_exp_sat = np.array([102.8,96.28,89.69,81.99,76.73,78.03,73.75,75.28]) #[C]


#Import data from CSV file
data = csv2rec('Cycle_60K_superheat_sub.csv',delimiter=',')
#Arrange data in Numpy array for the 8 different tests
m_dot = np.array([data[2][21],data[3][21],data[4][21],data[5][21],data[6][21],data[7][21],data[8][21],data[9][21]])
m_dot_inj = np.array([data[2][22],data[3][22],data[4][22],data[5][22],data[6][22],data[7][22],data[8][22],data[9][22]])
cooling_capacity = np.array([data[2][17],data[3][17],data[4][17],data[5][17],data[6][17],data[7][17],data[8][17],data[9][17]])
total_power = np.array([data[2][19],data[3][19],data[4][19],data[5][19],data[6][19],data[7][19],data[8][19],data[9][19]])
compressor_power = np.array([data[2][20],data[3][20],data[4][20],data[5][20],data[6][20],data[7][20],data[8][20],data[9][20]])
COPS = np.array([data[2][15],data[3][15],data[4][15],data[5][15],data[6][15],data[7][15],data[8][15],data[9][15]])
heating_capacity = np.array([data[2][16],data[3][16],data[4][16],data[5][16],data[6][16],data[7][16],data[8][16],data[9][16]])
PHX_capacity = np.array([data[2][18],data[3][18],data[4][18],data[5][18],data[6][18],data[7][18],data[8][18],data[9][18]])
charge = np.array([data[2][3],data[3][3],data[4][3],data[5][3],data[6][3],data[7][3],data[8][3],data[9][3]])
charge_corrected = np.array([data[2][4],data[3][4],data[4][4],data[5][4],data[6][4],data[7][4],data[8][4],data[9][4]])
charge_corrected_one = np.array([data[2][5],data[3][5],data[4][5],data[5][5],data[6][5],data[7][5],data[8][5],data[9][5]])
T_dis = np.array([data[2][57],data[3][57],data[4][57],data[5][57],data[6][57],data[7][57],data[8][57],data[9][57]])

#Import TUNED data from CSV file
data = csv2rec('Cycle_60K_superheat_sub_new.csv',delimiter=',')
#Arrange data in Numpy array for the 8 different tests
m_dot_tuned = np.array([data[2][21],data[3][21],data[4][21],data[5][21],data[6][21],data[7][21],data[8][21],data[9][21]])
m_dot_inj_tuned = np.array([data[2][22],data[3][22],data[4][22],data[5][22],data[6][22],data[7][22],data[8][22],data[9][22]])
cooling_capacity_tuned = np.array([data[2][17],data[3][17],data[4][17],data[5][17],data[6][17],data[7][17],data[8][17],data[9][17]])
total_power_tuned = np.array([data[2][19],data[3][19],data[4][19],data[5][19],data[6][19],data[7][19],data[8][19],data[9][19]])
compressor_power_tuned = np.array([data[2][20],data[3][20],data[4][20],data[5][20],data[6][20],data[7][20],data[8][20],data[9][20]])
COPS_tuned = np.array([data[2][15],data[3][15],data[4][15],data[5][15],data[6][15],data[7][15],data[8][15],data[9][15]])
heating_capacity_tuned = np.array([data[2][16],data[3][16],data[4][16],data[5][16],data[6][16],data[7][16],data[8][16],data[9][16]])
PHX_capacity_tuned = np.array([data[2][18],data[3][18],data[4][18],data[5][18],data[6][18],data[7][18],data[8][18],data[9][18]])
charge_tuned = np.array([data[2][3],data[3][3],data[4][3],data[5][3],data[6][3],data[7][3],data[8][3],data[9][3]])
charge_corrected_tuned = np.array([data[2][4],data[3][4],data[4][4],data[5][4],data[6][4],data[7][4],data[8][4],data[9][4]])
charge_corrected_one_tuned = np.array([data[2][5],data[3][5],data[4][5],data[5][5],data[6][5],data[7][5],data[8][5],data[9][5]])
T_dis_tuned = np.array([data[2][57],data[3][57],data[4][57],data[5][57],data[6][57],data[7][57],data[8][57],data[9][57]])

#Import SATURATED (UNTUNED) data from CSV file
data = csv2rec('Cycle_60K_saturated.csv',delimiter=',')
#Arrange data in Numpy array for the 8 different tests
m_dot_sat = np.array([data[2][21],data[3][21],data[4][21],data[5][21],data[6][21],data[7][21],data[8][21],data[9][21]])
m_dot_inj_sat = np.array([data[2][22],data[3][22],data[4][22],data[5][22],data[6][22],data[7][22],data[8][22],data[9][22]])
cooling_capacity_sat = np.array([data[2][17],data[3][17],data[4][17],data[5][17],data[6][17],data[7][17],data[8][17],data[9][17]])
total_power_sat = np.array([data[2][19],data[3][19],data[4][19],data[5][19],data[6][19],data[7][19],data[8][19],data[9][19]])
compressor_power_sat = np.array([data[2][20],data[3][20],data[4][20],data[5][20],data[6][20],data[7][20],data[8][20],data[9][20]])
COPS_sat = np.array([data[2][15],data[3][15],data[4][15],data[5][15],data[6][15],data[7][15],data[8][15],data[9][15]])
heating_capacity_sat = np.array([data[2][16],data[3][16],data[4][16],data[5][16],data[6][16],data[7][16],data[8][16],data[9][16]])
PHX_capacity_sat = np.array([data[2][18],data[3][18],data[4][18],data[5][18],data[6][18],data[7][18],data[8][18],data[9][18]])
charge_sat = np.array([data[2][3],data[3][3],data[4][3],data[5][3],data[6][3],data[7][3],data[8][3],data[9][3]])
charge_corrected_sat = np.array([data[2][4],data[3][4],data[4][4],data[5][4],data[6][4],data[7][4],data[8][4],data[9][4]])
charge_corrected_one_sat = np.array([data[2][5],data[3][5],data[4][5],data[5][5],data[6][5],data[7][5],data[8][5],data[9][5]])
T_dis_sat = np.array([data[2][57],data[3][57],data[4][57],data[5][57],data[6][57],data[7][57],data[8][57],data[9][57]])


#to convert string array to integer array
m_dot_sat = m_dot_sat.astype(np.float)
m_dot_inj_sat = m_dot_inj_sat.astype(np.float)
cooling_capacity_sat = cooling_capacity_sat.astype(np.float)
total_power_sat = total_power_sat.astype(np.float)
compressor_power_sat = compressor_power_sat.astype(np.float)
COPS_sat = COPS_sat.astype(np.float)
heating_capacity_sat = abs(heating_capacity_sat.astype(np.float))
PHX_capacity_sat = PHX_capacity_sat.astype(np.float)
charge_sat = charge_sat.astype(np.float)
charge_corrected_sat = charge_corrected_sat.astype(np.float)
charge_corrected_one_sat = charge_corrected_one_sat.astype(np.float)
T_dis_sat = T_dis_sat.astype(np.float) - 273.15 #convert from K to C

m_dot = m_dot.astype(np.float)
m_dot_inj = m_dot_inj.astype(np.float)
cooling_capacity = cooling_capacity.astype(np.float)
total_power = total_power.astype(np.float)
compressor_power = compressor_power.astype(np.float)
COPS = COPS.astype(np.float)
heating_capacity = abs(heating_capacity.astype(np.float))
PHX_capacity = PHX_capacity.astype(np.float)
charge = charge.astype(np.float)
charge_corrected = charge_corrected.astype(np.float)
charge_corrected_one = charge_corrected_one.astype(np.float)
T_dis = T_dis.astype(np.float) - 273.15 #convert from K to C

m_dot_tuned = m_dot_tuned.astype(np.float)
m_dot_inj_tuned = m_dot_inj_tuned.astype(np.float)
cooling_capacity_tuned = cooling_capacity_tuned.astype(np.float)
total_power_tuned = total_power_tuned.astype(np.float)
compressor_power_tuned = compressor_power_tuned.astype(np.float)
COPS_tuned = COPS_tuned.astype(np.float)
heating_capacity_tuned = abs(heating_capacity_tuned.astype(np.float))
PHX_capacity_tuned = PHX_capacity_tuned.astype(np.float)
charge_tuned = charge_tuned.astype(np.float)
charge_corrected_tuned = charge_corrected_tuned.astype(np.float)
charge_corrected_one_tuned = charge_corrected_one_tuned.astype(np.float)
T_dis_tuned = T_dis_tuned.astype(np.float) - 273.15 #convert from K to C

#plots
#Plot mass flow rate comparison
# plt.plot(TestNo,m_dot_exp*1000,'-ob',label='Experimental')
# plt.errorbar(TestNo,m_dot_exp*1000, yerr=0.001878*1000,fmt='',linestyle="None",color='k')
# plt.plot(TestNo,m_dot*1000,'--sr',label='Model (MAE = {:0.1f}%'.format(mape(m_dot,m_dot_exp))+', RMSD = {:0.1f}%)'.format(rmse(m_dot,m_dot_exp)))
# plt.plot(TestNo,m_dot_tuned*1000,':^g',label='Tuned (MAE = {:0.1f}%'.format(mape(m_dot_tuned,m_dot_exp))+', RMSD = {:0.1f}%)'.format(rmse(m_dot_tuned,m_dot_exp)))
# #plt.text(4,0.04,'MAE = {:0.01f}%'.format(mape(m_dot,m_dot_exp))+', RMSE = {:0.01f}%'.format(rmse(m_dot,m_dot_exp)),ha='left',va='center',fontsize = 10)
# plt.ylim(0.0,0.18*1000)
# plt.xlim(0,9)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.xlabel(r'Test condition')
# plt.ylabel(r'$\dot m_{suc}$ [g s$^{-1}$]')
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout()
# #plt.savefig('VI_massflow.pdf')
# #plt.show()
# plt.close()
# 
# #Plot injection mass flow rate comparison
# plt.plot(TestNo,m_dot_inj_exp*1000,'-ob',label='Experimental')
# plt.errorbar(TestNo,m_dot_inj_exp*1000, yerr=0.002902*1000,fmt='',linestyle="None",color='k')
# plt.plot(TestNo,m_dot_inj*1000,'--sr',label='Model (MAE = {:0.1f}%'.format(mape(m_dot_inj,m_dot_inj_exp))+', RMSD = {:0.1f}%)'.format(rmse(m_dot_inj,m_dot_inj_exp)))
# plt.plot(TestNo,m_dot_inj_tuned*1000,':^g',label='Tuned (MAE = {:0.1f}%'.format(mape(m_dot_inj_tuned,m_dot_inj_exp))+', RMSD = {:0.1f}%)'.format(rmse(m_dot_inj_tuned,m_dot_inj_exp)))
# #plt.text(4,0.03,'MAE = {:0.01f}%'.format(mape(m_dot_inj,m_dot_inj_exp))+', RMSE = {:0.01f}%'.format(rmse(m_dot_inj,m_dot_inj_exp)),ha='left',va='center',fontsize = 10)
# plt.ylim(0.0,0.04*1000)
# plt.xlim(0,9)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.xlabel(r'Test condition')
# plt.ylabel(r'$\dot m_{inj}$ [g s$^{-1}$]')
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout()
# #plt.savefig('VI_massflow_inj.pdf')
# #plt.show()
# plt.close()
# 
# #Plot Capacity comparison
# plt.plot(TestNo,cooling_capacity_exp,'-ob',label=r'Experimental')
# plt.errorbar(TestNo,cooling_capacity_exp, yerr=0.1679*cooling_capacity_exp,linestyle="None",color='k')
# plt.plot(TestNo,cooling_capacity/1000,'--sr',label='Model (MAE = {:0.1f}%'.format(mape(cooling_capacity/1000,cooling_capacity_exp))+', RMSD = {:0.1f}%)'.format(rmse(cooling_capacity/1000,cooling_capacity_exp)))
# plt.plot(TestNo,cooling_capacity_tuned/1000,':^g',label='Tuned (MAE = {:0.1f}%'.format(mape(cooling_capacity_tuned/1000,cooling_capacity_exp))+', RMSD = {:0.1f}%)'.format(rmse(cooling_capacity_tuned/1000,cooling_capacity_exp)))
# #plt.text(4,5,'MAE = {:0.01f}\%'.format(mape(cooling_capacity/1000,cooling_capacity_exp))+', RMSE = {:0.01f}\%'.format(rmse(cooling_capacity/1000,cooling_capacity_exp)),ha='left',va='center',fontsize = 10)
# plt.ylim(0,30)
# plt.xlim(0,9)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.xlabel(r'Test condition')
# plt.ylabel(r'$\dot Q_{evap}$ [kW]')
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout()
# plt.savefig('VI_capacity.pdf')
# plt.show()
# plt.close()
# 
# #Plot total power comparison
# plt.plot(TestNo,total_power_exp,'-ob',label='Experimental')
# plt.errorbar(TestNo,total_power_exp, yerr=0.2,linestyle="None",color='k')
# plt.plot(TestNo,total_power/1000,'--sr',label='Model (MAE = {:0.1f}%'.format(mape(total_power/1000,total_power_exp))+', RMSD = {:0.1f}%)'.format(rmse(total_power/1000,total_power_exp)))
# plt.plot(TestNo,total_power_tuned/1000,':^g',label='Tuned (MAE = {:0.1f}%'.format(mape(total_power_tuned/1000,total_power_exp))+', RMSD = {:0.1f}%)'.format(rmse(total_power_tuned/1000,total_power_exp)))
# #plt.text(4,2,'MAE = {:0.01f}\%'.format(mape(total_power/1000,total_power_exp))+', RMSE = {:0.01f}\%'.format(rmse(total_power/1000,total_power_exp)),ha='left',va='center',fontsize = 10)
# plt.ylim(0,12)
# plt.xlim(0,9)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.xlabel(r'Test condition')
# plt.ylabel(r'$\dot E_{tot}$ [kW]')
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout()
# plt.savefig('VI_total_power.pdf')
# plt.show()
# plt.close()
# 
# #Plot compressor power comparison
# plt.plot(TestNo,compressor_power_exp,'-ob',label='Experimental')
# plt.errorbar(TestNo,compressor_power_exp, yerr=0.1125,fmt='',linestyle="None",color='k')
# plt.plot(TestNo,compressor_power/1000,'--sr',label='Model (MAE = {:0.1f}%'.format(mape(compressor_power/1000,compressor_power_exp))+', RMSD = {:0.1f}%)'.format(rmse(compressor_power/1000,compressor_power_exp)))
# plt.plot(TestNo,compressor_power_tuned/1000,':^g',label='Tuned (MAE = {:0.1f}%'.format(mape(compressor_power_tuned/1000,compressor_power_exp))+', RMSD = {:0.1f}%)'.format(rmse(compressor_power_tuned/1000,compressor_power_exp)))
# #plt.text(4,2,'MAE = {:0.01f}%'.format(mape(compressor_power/1000,compressor_power_exp))+', RMSE = {:0.01f}%'.format(rmse(compressor_power/1000,compressor_power_exp)),ha='left',va='center',fontsize = 10)
# plt.ylim(0,10)
# plt.xlim(0,9)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.xlabel(r'Test condition')
# plt.ylabel(r'$\dot W_{comp}$ [kW]')
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout()
# #plt.savefig('VI_compressor_power.pdf')
# #plt.show()
# plt.close()
# 
# #Plot COPS comparison
# plt.plot(TestNo,COPS_exp,'-ob',label='Experimental')
# plt.errorbar(TestNo,COPS_exp, yerr=0.1704*COPS_exp,fmt='',linestyle="None",color='k')
# plt.plot(TestNo,COPS,'--sr',label='Model (MAE = {:0.1f}%'.format(mape(COPS,COPS_exp))+', RMSD = {:0.1f}%)'.format(rmse(COPS,COPS_exp)))
# plt.plot(TestNo,COPS_tuned,':^g',label='Tuned (MAE = {:0.1f}%'.format(mape(COPS_tuned,COPS_exp))+', RMSD = {:0.1f}%)'.format(rmse(COPS_tuned,COPS_exp)))
# #plt.text(4,1,'MAE = {:0.01f}%'.format(mape(COPS,COPS_exp))+', RMSE = {:0.01f}%'.format(rmse(COPS,COPS_exp)),ha='left',va='center',fontsize = 10)
# plt.ylim(0,5)
# plt.xlim(0,9)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.xlabel(r'Test condition')
# plt.ylabel(r'$\mathrm{COP}_{sys}$ [$-$]')
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout()
# #plt.savefig('VI_COPS.pdf')
# #plt.show()
# plt.close()
# 
# #Plot heating cooling_capacity comparison
# plt.plot(TestNo,heating_capacity_exp,'-ob',label=r'Experimental')
# plt.errorbar(TestNo,heating_capacity_exp, yerr=0.0228*heating_capacity_exp,linestyle="None",color='k')
# plt.plot(TestNo,heating_capacity/1000,'--sr',label='Model (MAE = {:0.1f}%'.format(mape(heating_capacity/1000,heating_capacity_exp))+', RMSD = {:0.1f}%)'.format(rmse(heating_capacity/1000,heating_capacity_exp)))
# plt.plot(TestNo,heating_capacity_tuned/1000,':^g',label='Tuned (MAE = {:0.1f}%'.format(mape(heating_capacity_tuned/1000,heating_capacity_exp))+', RMSD = {:0.1f}%)'.format(rmse(heating_capacity_tuned/1000,heating_capacity_exp)))
# #plt.text(4,5,'MAE = {:0.01f}\%'.format(mape(heating_capacity/1000,heating_capacity_exp))+', RMSE = {:0.01f}\%'.format(rmse(heating_capacity/1000,heating_capacity_exp)),ha='left',va='center',fontsize = 10)
# plt.ylim(0,45)
# plt.xlim(0,9)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.xlabel(r'Test condition')
# plt.ylabel(r'$\dot Q_{cond}$ [kW]')
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout()
# plt.savefig('VI_heatcapacity.pdf')
# plt.show()
# plt.close()
# 
# #Plot PHX cooling_capacity comparison
# plt.plot(TestNo,PHX_capacity_exp,'-ob',label=r'Experimental')
# plt.errorbar(TestNo,PHX_capacity_exp, yerr=0.0933*PHX_capacity_exp,linestyle="None",color='k')
# plt.plot(TestNo,PHX_capacity/1000,'--sr',label='Model (MAE = {:0.1f}%'.format(mape(PHX_capacity/1000,PHX_capacity_exp))+', RMSD = {:0.1f}%)'.format(rmse(PHX_capacity/1000,PHX_capacity_exp)))
# plt.plot(TestNo,PHX_capacity_tuned/1000,':^g',label='Tuned (MAE = {:0.1f}%'.format(mape(PHX_capacity_tuned/1000,PHX_capacity_exp))+', RMSD = {:0.1f}%)'.format(rmse(PHX_capacity_tuned/1000,PHX_capacity_exp)))
# #plt.text(4,1,'MAE = {:0.01f}\%'.format(mape(PHX_capacity/1000,PHX_capacity_exp))+', RMSE = {:0.01f}\%'.format(rmse(PHX_capacity/1000,PHX_capacity_exp)),ha='left',va='center',fontsize = 10)
# plt.ylim(0,8)
# plt.xlim(0,9)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.xlabel(r'Test condition')
# plt.ylabel(r'$\dot Q_{econ}$ [kW]')
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame = leg.get_frame()  
# frame.set_linewidth(0.5)
# plt.tight_layout()
# plt.savefig('VI_PHXcapacity.pdf')
# plt.show()
# plt.close()


#Plot MAE comparison for saturated and superheated injection
MAE_superheated = np.array([mape(T_dis,T_dis_exp), mape(m_dot,m_dot_exp), mape(m_dot_inj,m_dot_inj_exp), mape(compressor_power/1000,compressor_power_exp), mape(COPS,COPS_exp)])
MAE_saturated = np.array([mape(T_dis_sat,T_dis_exp_sat), mape(m_dot_sat,m_dot_exp_sat), mape(m_dot_inj_sat,m_dot_inj_exp_sat), mape(compressor_power_sat/1000,compressor_power_exp_sat), mape(COPS_sat,COPS_exp_sat)])
#print(MAE_superheated) #[  5.23501525   3.67057291  11.63221909   5.44809175   8.12792883]
#print(MAE_saturated) #[  5.02560604   4.34013514  17.53863102   5.59880961   9.22471872]

plt.bar(np.arange(1,6,1)-0.1,MAE_superheated,width=0.2,color='r',linewidth=0.9,align='center',alpha=0.9,label=r'Superheated',hatch=5*'\\')    
plt.bar(np.arange(1,6,1)+0.1,MAE_saturated,width=0.2,color='b',linewidth=0.9,align='center',alpha=0.9,label=r'Saturated',hatch=2*'//')

plt.ylim(0,20)
plt.xlim(0,6)
plt.xticks([0, 1, 2, 3, 4, 5, 6],
           [r'', r'$T_{dis}$', r'$\dot m_{suc}$', r'$\dot m_{inj}$',r'$\dot W_{comp}$', r'$\mathrm{COP}_{sys}$', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are on
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
#plt.xlabel(r'Test condition')
plt.ylabel(r'MAE [%]')
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout() 
plt.savefig('MAE.pdf')
plt.show()



#Plot charge comparison
# plt.plot(TestNo,charge_exp,'-ob',label='Experimental')
# plt.errorbar(TestNo,charge_exp, yerr=0.11,linestyle="None",color='k')
# plt.plot(TestNo,charge,'--sr',label='Model')
# plt.plot(TestNo,charge_corrected,':^k',label='Model Corrected - two points')
# plt.plot(TestNo,charge_corrected_one,'-.d',label='Model Corrected - one point')
# plt.text(4,4.8,'Two points: MAE = {:0.01f}\%'.format(mape(charge_corrected,charge_exp))+', RMSE = {:0.01f}\%'.format(rmse(charge_corrected,charge_exp)),ha='left',va='center',fontsize = 10)
# plt.text(4,4.75,'One point: MAE = {:0.01f}\%'.format(mape(charge_corrected_one,charge_exp))+', RMSE = {:0.01f}\%'.format(rmse(charge_corrected_one,charge_exp)),ha='left',va='center',fontsize = 10)
# plt.ylim(4.6,5.6)
# plt.xlim(0,9)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.xlabel(r'Test condition')
# plt.ylabel(r'$\mathrm{Charge}$ $[\mathrm{kg}]$')
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame = leg.get_frame()
# frame.set_linewidth(0.5)
# plt.tight_layout()
# plt.savefig('VI_charge.pdf')
# plt.show()
 
#Combine
# fig = plt.figure(1, figsize=(15,10))
# for i, gtype in enumerate(['Mass', 'Injection_Mass', 'Capacity', 'Power', 'Compressor', 'COPS','HeatCapacity','PHXCapacity','T_dis']):
#     ax = plt.subplot(3, 3, i+1)
#     if gtype.startswith('Mass'):
#         plt.plot(TestNo,m_dot_exp,'-ob',label='Experimental')
#         plt.errorbar(TestNo,m_dot_exp, yerr=0.001878)#0.002*m_dot_exp
#         plt.plot(TestNo,m_dot,'--sr',label='Model')
#         plt.text(4,0.04,'MAE = {:0.01f}\%'.format(mape(m_dot,m_dot_exp))+', RMSE = {:0.01f}\%'.format(rmse(m_dot,m_dot_exp)),ha='left',va='center',fontsize = 10)
#         plt.ylim(0.0,0.16)
#         plt.xlim(0,9)
#         plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                    [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
#         plt.xlabel(r'Test condition')
#         plt.ylabel(r'$\dot m_{suc}$ $[\mathrm{kg/s}]$')
#         leg = plt.legend(loc='best',fancybox=False,numpoints=1)
#         frame = leg.get_frame()  
#         frame.set_linewidth(0.5)
#         #plt.title('Mass flowrate Comparison')
#     if gtype.startswith('Injection_Mass'):
#         plt.plot(TestNo,m_dot_inj_exp,'-ob',label='Experimental')
#         plt.errorbar(TestNo,m_dot_inj_exp, yerr=0.002902)#0.002*m_dot_inj_exp
#         plt.plot(TestNo,m_dot_inj,'--sr',label='Model')
#         plt.text(4,0.03,'MAE = {:0.01f}\%'.format(mape(m_dot_inj,m_dot_inj_exp))+', RMSE = {:0.01f}\%'.format(rmse(m_dot_inj,m_dot_inj_exp)),ha='left',va='center',fontsize = 10)
#         plt.ylim(0.0,0.05)
#         plt.xlim(0,9)
#         plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                    [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
#         plt.xlabel(r'Test condition')
#         plt.ylabel(r'$\dot m_{inj}$ $[\mathrm{kg/s}]$')
#         leg = plt.legend(loc='best',fancybox=False,numpoints=1)
#         frame = leg.get_frame()  
#         frame.set_linewidth(0.5)
#         #plt.title('Mass flowrate Comparison')
#     if gtype.startswith('Capacity'):
#         plt.plot(TestNo,cooling_capacity_exp,'-ob',label='Experimental')
#         plt.errorbar(TestNo,cooling_capacity_exp, yerr=0.1679*cooling_capacity_exp)
#         plt.plot(TestNo,cooling_capacity/1000,'--sr',label='Model')
#         plt.text(4,5,'MAE = {:0.01f}\%'.format(mape(cooling_capacity/1000,cooling_capacity_exp))+', RMSE = {:0.01f}\%'.format(rmse(cooling_capacity/1000,cooling_capacity_exp)),ha='left',va='center',fontsize = 10)
#         plt.ylim(0,30)
#         plt.xlim(0,9)
#         plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                    [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
#         plt.xlabel(r'Test condition')
#         plt.ylabel(r'$\dot Q_{evap}$ $[\mathrm{kW}]$')
#         leg = plt.legend(loc='best',fancybox=False,numpoints=1)
#         frame = leg.get_frame()
#         frame.set_linewidth(0.5)
#         #plt.title('Capacity Comparison')
#     if gtype.startswith('T_dis'):
#         plt.plot(TestNo,T_dis_exp,'-ob',label='Experimental')
#         plt.errorbar(TestNo,T_dis_exp, yerr=1.1)
#         plt.plot(TestNo,T_dis,'--sr',label='Model')
#         plt.text(4,60,'MAE = {:0.01f}\%'.format(mape(T_dis,T_dis_exp))+', RMSE = {:0.01f}\%'.format(rmse(T_dis,T_dis_exp)),ha='left',va='center',fontsize = 10)
#         plt.ylim(50,120)
#         plt.xlim(0,9)
#         plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                    [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
#         plt.xlabel(r'Test condition')
#         plt.ylabel(r'$T_{dis}$ [{\textdegree}C]')
#         leg = plt.legend(loc='best',fancybox=False,numpoints=1)
#         frame = leg.get_frame()
#         frame.set_linewidth(0.5)
#         #plt.title('Discharge Temperature Comparison')
#     if gtype.startswith('Power'):
#         plt.plot(TestNo,total_power_exp,'-ob',label='Experimental')
#         plt.errorbar(TestNo,total_power_exp, yerr=0.2)#0.03*total_power_exp
#         plt.plot(TestNo,total_power/1000,'--sr',label='Model')
#         plt.text(4,2,'MAE = {:0.01f}\%'.format(mape(total_power/1000,total_power_exp))+', RMSE = {:0.01f}\%'.format(rmse(total_power/1000,total_power_exp)),ha='left',va='center',fontsize = 10)
#         plt.ylim(0,12)
#         plt.xlim(0,9)
#         plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                    [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
#         plt.xlabel(r'Test condition')
#         plt.ylabel(r'$\dot E_t$ $[\mathrm{kW}]$')
#         leg = plt.legend(loc='best',fancybox=False,numpoints=1)
#         frame = leg.get_frame()
#         frame.set_linewidth(0.5)
#         #plt.title('Total Power Comparison')
#     if gtype.startswith('Compressor'):
#         plt.plot(TestNo,compressor_power_exp,'-ob',label='Experimental')
#         plt.errorbar(TestNo,compressor_power_exp, yerr=0.1125)
#         plt.plot(TestNo,compressor_power/1000,'--sr',label='Model')
#         plt.text(4,2,'MAE = {:0.01f}\%'.format(mape(compressor_power/1000,compressor_power_exp))+', RMSE = {:0.01f}\%'.format(rmse(compressor_power/1000,compressor_power_exp)),ha='left',va='center',fontsize = 10)
#         plt.ylim(0,10)
#         plt.xlim(0,9)
#         plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                    [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
#         plt.xlabel(r'Test condition')
#         plt.ylabel(r'$\dot W_{comp}$ $[\mathrm{kW}]$')
#         leg = plt.legend(loc='best',fancybox=False,numpoints=1)
#         frame = leg.get_frame()
#         frame.set_linewidth(0.5)
#         #plt.title('Compressor Power Comparison')
#     if gtype.startswith('COPS'):
#         plt.plot(TestNo,COPS_exp,'-ob',label='Experimental')
#         plt.errorbar(TestNo,COPS_exp, yerr=0.1704*COPS_exp)
#         plt.plot(TestNo,COPS,'--sr',label='Model')
#         plt.text(4,1,'MAE = {:0.01f}\%'.format(mape(COPS,COPS_exp))+', RMSE = {:0.01f}\%'.format(rmse(COPS,COPS_exp)),ha='left',va='center',fontsize = 10)
#         plt.ylim(0,5)
#         plt.xlim(0,9)
#         plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                    [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
#         plt.xlabel(r'Test condition')
#         plt.ylabel(r'$\mathrm{COP}_{sys}$')
#         leg = plt.legend(loc='best',fancybox=False,numpoints=1)
#         frame = leg.get_frame()
#         frame.set_linewidth(0.5)
#         #plt.title('System COP Comparison')
#     if gtype.startswith('HeatCapacity'):
#         plt.plot(TestNo,heating_capacity_exp,'-ob',label='Experimental')
#         plt.errorbar(TestNo,heating_capacity_exp, yerr=0.0228*heating_capacity_exp)
#         plt.plot(TestNo,heating_capacity/1000,'--sr',label='Model')
#         plt.text(4,10,'MAE = {:0.01f}\%'.format(mape(heating_capacity/1000,heating_capacity_exp))+', RMSE = {:0.01f}\%'.format(rmse(heating_capacity/1000,heating_capacity_exp)),ha='left',va='center',fontsize = 10)
#         plt.ylim(0,45)
#         plt.xlim(0,9)
#         plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                    [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
#         plt.xlabel(r'Test condition')
#         plt.ylabel(r'$\dot Q_{cond}$ $[\mathrm{kW}]$')
#         leg = plt.legend(loc='best',fancybox=False,numpoints=1)
#         frame = leg.get_frame()
#         frame.set_linewidth(0.5)
#         #plt.title('Heating Capacity Comparison')
#     if gtype.startswith('PHXCapacity'):
#         plt.plot(TestNo,PHX_capacity_exp,'-ob',label='Experimental')
#         plt.errorbar(TestNo,PHX_capacity_exp, yerr=0.0933*PHX_capacity_exp)
#         plt.plot(TestNo,PHX_capacity/1000,'--sr',label='Model')
#         plt.text(4,1,'MAE = {:0.01f}\%'.format(mape(PHX_capacity/1000,PHX_capacity_exp))+', RMSE = {:0.01f}\%'.format(rmse(PHX_capacity/1000,PHX_capacity_exp)),ha='left',va='center',fontsize = 10)
#         plt.ylim(0,8)
#         plt.xlim(0,9)
#         plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                    [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
#         plt.xlabel(r'Test condition')
#         plt.ylabel(r'$\dot Q_{econ}$ $[\mathrm{kW}]$')
#         leg = plt.legend(loc='best',fancybox=False,numpoints=1)
#         frame = leg.get_frame()
#         frame.set_linewidth(0.5)
#         #plt.title('PHX Capacity Comparison')
#     if gtype.startswith('Charge'):
#         plt.plot(TestNo,charge_exp,'-ob',label='Experimental')
#         plt.errorbar(TestNo,charge_exp, yerr=0.11)
#         plt.plot(TestNo,charge,'--sr',label='Model')
#         plt.plot(TestNo,charge_corrected,':^k',label='Model Corrected - two points')
#         plt.plot(TestNo,charge_corrected_one,'-.d',label='Model Corrected - one point')
#         plt.text(3,4.8,'Two points: MAE = {:0.01f}\%'.format(mape(charge_corrected,charge_exp))+', RMSE = {:0.01f}\%'.format(rmse(charge_corrected,charge_exp)),ha='left',va='center',fontsize = 10)
#         plt.text(3,4.75,'One point: MAE = {:0.01f}\%'.format(mape(charge_corrected_one,charge_exp))+', RMSE = {:0.01f}\%'.format(rmse(charge_corrected_one,charge_exp)),ha='left',va='center',fontsize = 10)
#         plt.ylim(4.6,5.6)
#         plt.xlim(0,9)
#         plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                    [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
#         plt.xlabel(r'Test condition')
#         plt.ylabel(r'$\mathrm{Charge}$ $[\mathrm{kg}]$')
#         leg = plt.legend(loc='best',fancybox=False,numpoints=1)
#         frame = leg.get_frame()
#         frame.set_linewidth(0.5)
#         #plt.title('System charge Comparison')
# fig.set_tight_layout(True)
# plt.savefig('VI_comined_sub.pdf')
# plt.show()
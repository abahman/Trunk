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

#Experimental Results
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

#Import data from CSV file
data = csv2rec('Cycle_60K_superheat_sub_new.csv',delimiter=',')
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

#to convert string array to integer array
m_dot = m_dot.astype(np.float)
m_dot_inj = m_dot_inj.astype(np.float)
cooling_capacity = cooling_capacity.astype(np.float)/1000
total_power = total_power.astype(np.float)/1000
compressor_power = compressor_power.astype(np.float)/1000
COPS = COPS.astype(np.float)
heating_capacity = abs(heating_capacity.astype(np.float))/1000
PHX_capacity = PHX_capacity.astype(np.float)/1000
charge = charge.astype(np.float)
charge_corrected = charge_corrected.astype(np.float)
charge_corrected_one = charge_corrected_one.astype(np.float)
T_dis = T_dis.astype(np.float) - 273.15 #convert from K to C

#average experimnetal data
m_dot_mean = np.mean(m_dot_exp) #[kg/s]
m_dot_tot_mean = np.mean(m_dot_tot_exp) #[kg/s]
m_dot_inj_mean = np.mean(m_dot_inj_exp) #[kg/s]
cooling_capacity_mean = np.mean(cooling_capacity_exp) #[kW]
total_power_mean = np.mean(total_power_exp) #[kW]
compressor_power_mean = np.mean(compressor_power_exp) #[kW]
COPS_mean = np.mean(COPS_exp) #[-]
charge_mean = np.mean(charge_exp) #[kg]
heating_capacity_mean = np.mean(heating_capacity_exp) #[kW]
PHX_capacity_mean = np.mean(PHX_capacity_exp) #[kW]
T_dis_mean = np.mean(T_dis_exp) #[C]


#########################
#####Combined plot#######
#########################
f,ax=plt.subplots(figsize=(4,4))

w=0.1 #Error
ax_max = 2.5 #x and y-axes max scale tick
upp_txt = ax_max / 2 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = ax_max / 2 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'$-${:0.0f}%'.format(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}%'.format(w*100),ha='right',va='bottom')
ax.set_xlabel('Normalized experiment value')
ax.set_ylabel('Normalized model value')

ax.plot(T_dis_exp/T_dis_mean,T_dis/T_dis_mean,'o',ms=4,markerfacecolor='None',label='$T_{{dis}}$ (MAE = {:0.1f}%'.format(mape(T_dis/T_dis_mean,T_dis_exp/T_dis_mean))+', RMSD = {:0.1f}%)'.format(rmse(T_dis/T_dis_mean,T_dis_exp/T_dis_mean)),mec='blue',mew=1)
ax.plot(m_dot_exp/m_dot_mean,m_dot/m_dot_mean,'s',ms=4,markerfacecolor='None',label='$\dot m_{{suc}}$ (MAE = {:0.1f}%'.format(mape(m_dot/m_dot_mean,m_dot_exp/m_dot_mean))+', RMSD = {:0.1f}%)'.format(rmse(m_dot/m_dot_mean,m_dot_exp/m_dot_mean)),mec='red',mew=1)
ax.plot(m_dot_inj_exp/m_dot_inj_mean,m_dot_inj/m_dot_inj_mean,'^',ms=4,markerfacecolor='None',label='$\dot m_{{inj}}$ (MAE = {:0.1f}%'.format(mape(m_dot_inj/m_dot_inj_mean,m_dot_inj_exp/m_dot_inj_mean))+', RMSD = {:0.1f}%)'.format(rmse(m_dot_inj/m_dot_inj_mean,m_dot_inj_exp/m_dot_inj_mean)),mec='green',mew=1)
#ax.plot(cooling_capacity_exp/cooling_capacity_mean,cooling_capacity/cooling_capacity_mean,'H',ms=4,markerfacecolor='None',label='$\dot Q_{{evap}}$ (MAE = {:0.1f}%'.format(mape(cooling_capacity/cooling_capacity_mean,cooling_capacity_exp/cooling_capacity_mean))+', RMSD = {:0.1f}%)'.format(rmse(cooling_capacity/cooling_capacity_mean,cooling_capacity_exp/cooling_capacity_mean)),mec='c',mew=1)
ax.plot(compressor_power_exp/compressor_power_mean,compressor_power/compressor_power_mean,'*',ms=5,markerfacecolor='None',label='$\dot W_{{comp}}$ (MAE = {:0.1f}%'.format(mape(compressor_power/compressor_power_mean,compressor_power_exp/compressor_power_mean))+', RMSD = {:0.1f}%)'.format(rmse(compressor_power/compressor_power_mean,compressor_power_exp/compressor_power_mean)),mec='brown',mew=1)
ax.plot(COPS_exp/COPS_mean,COPS/COPS_mean,'D',ms=4,markerfacecolor='None',label='COP$_{{sys}}$ (MAE = {:0.1f}%'.format(mape(COPS/COPS_mean,COPS_exp/COPS_mean))+', RMSD = {:0.1f}%)'.format(rmse(COPS/COPS_mean,COPS_exp/COPS_mean)),mec='orange',mew=1)


leg=ax.legend(loc='upper left',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
ax.set_xlim((0,ax_max))
ax.set_ylim((0,ax_max))
plt.tight_layout()
plt.savefig('VI_combined_tuned.pdf')
plt.show()
plt.close()
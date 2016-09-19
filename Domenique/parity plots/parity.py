import matplotlib,os
#matplotlib.use('GTKAgg')
import sys, os
#from FileIO import prep_csv2rec as prep
from matplotlib.mlab import csv2rec

import pylab
import numpy as np
import shutil
from scipy import polyval, polyfit
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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


def rmse(predictions, targets):
    '''
    Root Mean Square Error
    '''
    n = len(predictions)
    RMSE = np.linalg.norm(predictions - targets) / np.sqrt(n)
    return RMSE

def mape(y_pred, y_true):  #maps==mean_absolute_percentage_error
    '''
    Mean Absolute Percentage Error
    '''
    MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return MAPE


#import data from excel file
df = pd.read_excel('dimensionless.xlsx') #file name

#experimental data
m_dot_inj_norm_exp = np.array(df[1:]['m_dot_inj_norm'], dtype=float)
m_dot_suc_exp = np.array(df[1:]['m_dot_suc'], dtype=float)
m_dot_inj_exp = np.array(df[1:]['m_dot_inj'], dtype=float)
m_dot_total_exp = np.array(df[1:]['m_dot_total'], dtype=float)
P_exp = np.array(df[1:]['P'], dtype=float)
T_dis_exp = np.array(df[1:]['T_dis'], dtype=float)

T_sat_suc = np.array(df[1:]['T_evap_stpt'], dtype=float)

#yuanpei correlation
yuanpei_m_dot_inj_norm = np.array(df[1:]['yuanpei_m_dot_inj_norm'], dtype=float)
yuanpei_m_dot_suc = np.array(df[1:]['yuanpei_m_dot_suc'], dtype=float)
yuanpei_m_dot_inj = np.array(df[1:]['yuanpei_m_dot_inj'], dtype=float)
yuanpei_m_dot_total = np.array(df[1:]['yuanpei_m_dot_total'], dtype=float)
yuanpei_p = np.array(df[1:]['yuanpei_p'], dtype=float)
yaunpei_t_dis = np.array(df[1:]['yaunpei_t_dis'], dtype=float)

#my correlation A and B
my_m_dot_inj_norm_A = np.array(df[1:]['my_m_dot_inj_norm_A'], dtype=float)
my_m_dot_inj_norm_B = np.array(df[1:]['my_m_dot_inj_norm_B'], dtype=float)
my_m_dot_suc_A = np.array(df[1:]['my_m_dot_suc_A'], dtype=float)
my_m_dot_suc_B = np.array(df[1:]['my_m_dot_suc_B'], dtype=float)
my_p_A = np.array(df[1:]['my_p_A'], dtype=float)
my_p_B = np.array(df[1:]['my_p_B'], dtype=float)
my_t_dis_A = np.array(df[1:]['my_t_dis_A'], dtype=float)
my_t_dis_B = np.array(df[1:]['my_t_dis_B'], dtype=float)
my_m_dot_inj_A = np.array(df[1:]['my_m_dot_inj_A'], dtype=float)
my_m_dot_inj_B = np.array(df[1:]['my_m_dot_inj_B'], dtype=float)
my_m_dot_total_A = np.array(df[1:]['my_m_dot_total_A'], dtype=float)
my_m_dot_total_B = np.array(df[1:]['my_m_dot_total_B'], dtype=float)

#domenique correlation
dom_m_dot_inj_norm = np.array(df[1:]['dom_m_dot_inj_norm'], dtype=float)
dom_m_dot_suc = np.array(df[1:]['dom_m_dot_suc'], dtype=float)
dom_p = np.array(df[1:]['dom_p'], dtype=float)
dom_t_dis = np.array(df[1:]['dom_t_dis'], dtype=float)
dom_m_dot_inj = np.array(df[1:]['dom_m_dot_inj'], dtype=float)
dom_m_dot_total = np.array(df[1:]['dom_m_dot_total'], dtype=float)

######RMSE#######
rmse_mass = rmse(yuanpei_m_dot_inj_norm,m_dot_inj_norm_exp)
print("rmse_mass error is: " + str(rmse_mass*100) + " %")
######MAPE######
mape_mass = mape(yuanpei_m_dot_inj_norm,m_dot_inj_norm_exp)
print("mape_mass error is: " + str(mape_mass) + " %")


#########################
#####Combined plot#######
#########################
#assign axes
y1 = yuanpei_m_dot_inj_norm
y2 = my_m_dot_inj_norm_A
y3 = my_m_dot_inj_norm_B
y4 = dom_m_dot_inj_norm
x = m_dot_inj_norm_exp
c = T_sat_suc
s = 20  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y1, c=c, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y1,x)))
im = ax.scatter(x, y2, c=c, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y2,x)))
im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y3,x)))
im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y4,x)))
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(1000, 2600)
cbar.ax.set_ylabel('Sat. suction temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
#plt.ylim(80,100)
#plt.xlim(0,24.875)

#error axes
w=mape_mass/100 #Error
ax_max = 0.5 #x and y-axes max scale tick
upp_txt = ax_max / 1.8 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = ax_max / 1.8 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'-{:0.0f}\%'.format(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}\%'.format(w*100),ha='right',va='bottom')
leg=ax.legend(loc='upper left',numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
ax.set_xlim((0,ax_max))
ax.set_ylim((0,ax_max))

plt.ylabel('$\dot m_{inj}$/$\dot m_{suc}$ predicted [\%]')
plt.xlabel('$\dot m_{inj}$/$\dot m_{suc}$ measured [\%]')           
plt.savefig('parity_m_inj_norm.pdf')
plt.show()
plt.close()

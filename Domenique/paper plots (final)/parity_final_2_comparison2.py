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


    ######RMSE#######
    #rmse_mass = rmse(yuanpei_m_dot_inj_norm,m_dot_inj_norm_exp)
    #print("rmse_mass error is: " + str(rmse_mass) + " %")
    ######MAPE######
    # mape_mass = mape(yuanpei_m_dot_inj_norm,m_dot_inj_norm_exp)
    # print("mape_mass error is: " + str(mape_mass) + " %")
    
    

#########################
##### m_inj/m_suc #######
#########################
#import data from excel file
df_total = pd.read_excel('correlation_results.xlsx',sheetname='Total_injection_rate_update',header=0) #file name
df_vapor = pd.read_excel('correlation_results.xlsx',sheetname='Vapor_injection_rate_update',header=0) #file name
df_two_phase = pd.read_excel('correlation_results.xlsx',sheetname='Two_phase_injection_rate_update',header=0) #file name
df_dardenne = pd.read_excel('correlation_results.xlsx',sheetname='Dardenne_injection_rate_update',header=0) #file name

#assign axes
y1 = df_total['InjectionRatio_pred[i]'][1:]*100
y2 = df_vapor['InjectionRatio_pred[i]'][1:]*100
y3 = df_two_phase['InjectionRatio_pred[i]'][1:]*100
y4 = df_dardenne['InjectionRatio_pred[i]'][1:]*100

x1 = df_total['InjectionRatio[i]'][1:]*100
x2 = df_vapor['InjectionRatio[i]'][1:]*100
x3 = df_two_phase['InjectionRatio[i]'][1:]*100
x4 = df_dardenne['InjectionRatio[i]'][1:]*100

#c1 = (df['T_evap_stpt_ENG[i]'][1:] - 32.0) * 5.0/9.0 + 273.15
#c2 = df['T_evap[i]'][1:]
s = 20  # size of points
  
fig, ax = plt.subplots(figsize=(4.5,4.5))
im = ax.scatter(x1, y1, c='b', s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Combined data'+' (MAE = {:0.01f}\%'.format(mape(y1,x1))+', RMSE = {:0.01f}\%)'.format(rmse(y1,x1)))
im = ax.scatter(x2, y2, c='r', s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='Vapor only'+' (MAE = {:0.01f}\%'.format(mape(y2,x2))+', RMSE = {:0.01f}\%)'.format(rmse(y2,x2)))
im = ax.scatter(x3, y3, c='k', s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Two-phase only'+' (MAE = {:0.01f}\%'.format(mape(y3,x3))+', RMSE = {:0.01f}\%)'.format(rmse(y3,x3)))
im = ax.scatter(x4, y4, c='g', s=s, cmap=plt.cm.jet, marker='d',lw=0.2, alpha =0.9,label='Dardenne data'+' (MAE = {:0.01f}\%'.format(mape(y4,x4))+', RMSE = {:0.01f}\%)'.format(rmse(y4,x4)))
#im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y3,x)))
#im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y4,x)))
# Add a colorbar
#cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(245, 290)
#cbar.ax.set_ylabel('Evaporation temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
  
#error axes
w=0.12 #Error
ax_min = 0
ax_max = 80 #x and y-axes max scale tick
upp_txt = (ax_min+ax_max) / 1.8 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = (ax_min+ax_max) / 1.25 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'$-${:0.0f}\%'.format(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}\%'.format(w*100),ha='right',va='bottom')
leg=ax.legend(loc='upper left',scatterpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
ax.set_xlim((ax_min,ax_max))
ax.set_ylim((ax_min,ax_max))
plt.ylabel('$\dot m_{inj}$/$\dot m_{suc}$ predicted [\%]')
plt.xlabel('$\dot m_{inj}$/$\dot m_{suc}$ measured [\%]')
plt.tight_layout()       
plt.savefig('All_parity_m_inj_comparison_updated.pdf')
plt.show()
plt.close()

# #########################
# ##### work (power) ######
# #########################
# #import data from excel file
# df_total = pd.read_excel('correlation_results.xlsx',sheetname='Total_work',header=0) #file name
# df_vapor = pd.read_excel('correlation_results.xlsx',sheetname='Vapor_work',header=0) #file name
# df_two_phase = pd.read_excel('correlation_results.xlsx',sheetname='Two_phase_work',header=0) #file name
# df_dardenne = pd.read_excel('correlation_results.xlsx',sheetname='Dardenne_work',header=0) #file name#assign axes
# 
# #assign axes
# y1 = df_total['WT1_P_UUT_pred[i]'][1:] #Watts
# y2 = df_vapor['WT1_P_UUT_pred[i]'][1:] #Watts
# y3 = df_two_phase['WT1_P_UUT_pred[i]'][1:] #Watts
# y4 = df_dardenne['W_dot_pred[i]'][1:] #Watts
# 
# x1 = df_total['WT1_P_UUT[i]'][1:] #watts
# x2 = df_vapor['WT1_P_UUT[i]'][1:] #watts
# x3 = df_two_phase['WT1_P_UUT[i]'][1:] #watts
# x4 = df_dardenne['W_dot[i]'][1:] #watts
# 
# #c1 = (df['T_evap_stpt_ENG[i]'][1:] - 32.0) * 5.0/9.0 + 273.15
# #c2 = df_dar['T_evap[i]'][1:]
# s = 20  # size of points
#   
# fig, ax = plt.subplots(figsize=(4.5,4.5))
# im = ax.scatter(x1, y1, c='b', s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Combined data'+' (MAE = {:0.01f}\%'.format(mape(y1,x1))+', RMSE = {:0.01f}\%)'.format(rmse(y1,x1)))
# im = ax.scatter(x2, y2, c='r', s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='Vapor only'+' (MAE = {:0.01f}\%'.format(mape(y2,x2))+', RMSE = {:0.01f}\%)'.format(rmse(y2,x2)))
# im = ax.scatter(x3, y3, c='k', s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Two-phase only'+' (MAE = {:0.01f}\%'.format(mape(y3,x3))+', RMSE = {:0.01f}\%)'.format(rmse(y3,x3)))
# im = ax.scatter(x4, y4, c='g', s=s, cmap=plt.cm.jet, marker='d',lw=0.2, alpha =0.9,label='Dardenne data'+' (MAE = {:0.01f}\%'.format(mape(y4,x4))+', RMSE = {:0.01f}\%)'.format(rmse(y4,x4)))
# #im = ax.scatter(x2, y11, c=c2, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}\%'.format(mape(y11,x2))+', RMSE = {:0.01f}\%)'.format(rmse(y11,x2)))
# #im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y3,x)))
# #im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y4,x)))
# # Add a colorbar
# #cbar = plt.colorbar(im, ax=ax)
# # set the color limits
# #im.set_clim(245, 290)
# #cbar.ax.set_ylabel('Evaporation temperature [K]')
# #ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
#   
# #error axes
# w=0.05 #Error
# ax_min = 0
# ax_max = 9000 #x and y-axes max scale tick
# upp_txt = (ax_min+ax_max) / 1.9 #location of upper error text on plot -- adjust the number to adjust the location
# low_txt = (ax_min+ax_max) / 1.9 #location of lower error text on plot -- adjust the number to adjust the location
# ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
# ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
# ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
# ax.text(low_txt-0.002,low_txt*(1-w),'$-${:0.0f}\%'.format(w*100),ha='left',va='top')
# ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}\%'.format(w*100),ha='right',va='bottom')
# leg=ax.legend(loc='upper left',scatterpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
# ax.set_xlim((ax_min,ax_max))
# ax.set_ylim((ax_min,ax_max))
# plt.ylabel('$\dot W$ predicted [W]')
# plt.xlabel('$\dot W$ measured [W]')
# plt.tight_layout()       
# #plt.savefig('All_parity_work_comparison.pdf')
# #plt.show()
# plt.close()  
       

 
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
    RMSE = np.linalg.norm(predictions - targets) / np.sqrt(n) / np.mean(targets) * 100
    return RMSE

def mape(y_pred, y_true):  #maps==mean_absolute_percentage_error
    '''
    Mean Absolute Percentage Error
    '''
    MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return MAPE


#import data from excel file
df = pd.read_excel('data_final.xlsx') #file name
df_dar = pd.read_excel('Dardenne.xlsx') #file name

#experimental data
T_evap = np.array(df[0:]['T_evap [K]'], dtype=float)
T_dis_exp = np.array(df[0:]['Actual Discharge Temperature (K)'], dtype=float)
T_dis_corr = np.array(df[0:]['Predicted Discharge Temperature (K)'], dtype=float)
T_dis_ARI = np.array(df[0:]['T_dis_ARI'], dtype=float)

m_ratio_exp = np.array(df[0:]['Actual Injection Ratio'], dtype=float) * 100
m_ratio_corr = np.array(df[0:]['Predicted Injection Ratio'], dtype=float) * 100
m_ratio_ARI = np.array(df[0:]['m_ratio_ARI'], dtype=float) * 100

eta_c_exp = np.array(df[0:]['Actual Isentropic Efficiency'], dtype=float) * 100
eta_c_corr = np.array(df[0:]['Predicted Isentropic Efficiency'], dtype=float) * 100
eta_c_ARI = np.array(df[0:]['eta_isen_ARI'], dtype=float) * 100

eta_v_exp = np.array(df[0:]['Actual Volumetric Efficiency'], dtype=float) * 100
eta_v_corr = np.array(df[0:]['Predicted Volumetric Efficiency'], dtype=float) * 100
eta_v_ARI = np.array(df[0:]['eta_v_ARI'], dtype=float) * 100

f_q_exp = np.array(df[0:]['actual heat loss'], dtype=float)
f_q_corr1 = np.array(df[0:]['predicted heat loss (w/o T_amb)'], dtype=float)
f_q_corr2 = np.array(df[0:]['predicted heat loss (w/ T_amb)'], dtype=float)
f_q_ARI = np.array(df[0:]['f_loss_ARI'], dtype=float)

#Dardenne data
T_evap_dar = np.array(df_dar[0:]['T_evap [K]'], dtype=float)
T_dis_exp_dar = np.array(df_dar[0:]['Actual Discharge Temperature (K)'], dtype=float)
T_dis_corr_dar = np.array(df_dar[0:]['Predicted Discharge Temperature (K)'], dtype=float)
m_ratio_exp_dar = np.array(df_dar[0:]['Actual injection rate'], dtype=float) * 100
m_ratio_corr_dar = np.array(df_dar[0:]['Predicted injection rate'], dtype=float) * 100
eta_c_exp_dar = np.array(df_dar[0:]['Actual Isentropic Efficiency'], dtype=float) * 100
eta_c_corr_dar = np.array(df_dar[0:]['Predicted Isentropic Efficiency'], dtype=float) * 100
eta_v_exp_dar = np.array(df_dar[0:]['Actual Volumetric Efficiency'], dtype=float) * 100
eta_v_corr_dar = np.array(df_dar[0:]['Predicted Volumetric Efficiency'], dtype=float) * 100
f_q_exp_dar = np.array(df_dar[0:]['Actual Heat Loss'], dtype=float)
f_q_corr_dar = np.array(df_dar[0:]['Predicted Heat Loss'], dtype=float)

# 
# #my correlation A and B
# my_m_dot_inj_norm_A = np.array(df[1:]['my_m_dot_inj_norm_A'], dtype=float) * 100
# my_m_dot_inj_norm_B = np.array(df[1:]['my_m_dot_inj_norm_B'], dtype=float) * 100
# my_m_dot_suc_A = np.array(df[1:]['my_m_dot_suc_A'], dtype=float) * 0.453592 #convert lbm/hr to kg/hr
# my_m_dot_suc_B = np.array(df[1:]['my_m_dot_suc_B'], dtype=float) * 0.453592 #convert lbm/hr to kg/hr
# my_p_A = np.array(df[1:]['my_p_A'], dtype=float) /1000.0 #convert W to kW
# my_p_B = np.array(df[1:]['my_p_B'], dtype=float) /1000.0 #convert W to kW
# my_t_dis_A = (np.array(df[1:]['my_t_dis_A'], dtype=float) + 459.67) * 5.0/9.0  #convert F to K
# my_t_dis_B = (np.array(df[1:]['my_t_dis_B'], dtype=float) + 459.67) * 5.0/9.0  #convert F to K
# my_m_dot_inj_A = np.array(df[1:]['my_m_dot_inj_A'], dtype=float) * 0.453592 #convert lbm/hr to kg/hr
# my_m_dot_inj_B = np.array(df[1:]['my_m_dot_inj_B'], dtype=float) * 0.453592 #convert lbm/hr to kg/hr
# my_m_dot_total_A = np.array(df[1:]['my_m_dot_total_A'], dtype=float) * 0.453592 #convert lbm/hr to kg/hr
# my_m_dot_total_B = np.array(df[1:]['my_m_dot_total_B'], dtype=float) * 0.453592 #convert lbm/hr to kg/hr
# 
# #domenique correlation
# dom_m_dot_inj_norm = np.array(df[1:]['dom_m_dot_inj_norm'], dtype=float) * 100
# dom_m_dot_suc = np.array(df[1:]['dom_m_dot_suc'], dtype=float) * 0.453592 #convert lbm/hr to kg/hr
# dom_p = np.array(df[1:]['dom_p'], dtype=float) /1000.0 #convert W to kW
# dom_t_dis = (np.array(df[1:]['dom_t_dis'], dtype=float) + 459.67) * 5.0/9.0  #convert F to K
# dom_m_dot_inj = np.array(df[1:]['dom_m_dot_inj'], dtype=float) * 0.453592 #convert lbm/hr to kg/hr
# dom_m_dot_total = np.array(df[1:]['dom_m_dot_total'], dtype=float) * 0.453592 #convert lbm/hr to kg/hr

######RMSE#######
#rmse_mass = rmse(yuanpei_m_dot_inj_norm,m_dot_inj_norm_exp)
#print("rmse_mass error is: " + str(rmse_mass) + " %")
######MAPE######
# mape_mass = mape(yuanpei_m_dot_inj_norm,m_dot_inj_norm_exp)
# print("mape_mass error is: " + str(mape_mass) + " %")


#########################
##### m_inj/m_suc #######
#########################
#assign axes
y1 = m_ratio_corr
y2 = m_ratio_corr_dar
y11 = m_ratio_ARI
#y3 = my_m_dot_inj_norm_B
#y4 = dom_m_dot_inj_norm
x1 = m_ratio_exp
x2 = m_ratio_exp_dar
c1 = T_evap
c2 = T_evap_dar
s = 20  # size of points
 
fig, ax = plt.subplots()
im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}\%'.format(mape(y1,x1))+', RMSE = {:0.01f}\%)'.format(rmse(y1,x1)))
im = ax.scatter(x1, y11, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='ARI'+' (MAE = {:0.01f}\%'.format(mape(y11,x1))+', RMSE = {:0.01f}\%)'.format(rmse(y11,x1)))
im = ax.scatter(x2, y2, c=c2, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}\%'.format(mape(y2,x2))+', RMSE = {:0.01f}\%)'.format(rmse(y2,x2)))
#im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y3,x)))
#im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y4,x)))
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(245, 290)
cbar.ax.set_ylabel('Evaporation temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
 
 
#error axes
w=0.1 #Error
ax_min = 0
ax_max = 65 #x and y-axes max scale tick
upp_txt = (ax_min+ax_max) / 2.05 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = (ax_min+ax_max) / 2.0 #location of lower error text on plot -- adjust the number to adjust the location
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
plt.savefig('parity_m_inj.pdf')
#plt.show()
plt.close()
  
  
#########################
##### T_dis  #######
#########################
#assign axes
y1 = T_dis_corr
y2 = T_dis_corr_dar
y11 = T_dis_ARI
#y3 = my_t_dis_B
#y4 = dom_t_dis
x1 = T_dis_exp
x2 = T_dis_exp_dar
c1 = T_evap
c2 = T_evap_dar
s = 20  # size of points
  
fig, ax = plt.subplots()
im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}\%'.format(mape(y1,x1))+', RMSE = {:0.01f}\%)'.format(rmse(y1,x1)))
im = ax.scatter(x1, y11, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='ARI'+' (MAE = {:0.01f}\%'.format(mape(y11,x1))+', RMSE = {:0.01f}\%)'.format(rmse(y11,x1)))
im = ax.scatter(x2, y2, c=c2, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}\%'.format(mape(y2,x2))+', RMSE = {:0.01f}\%)'.format(rmse(y2,x2)))
#im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y3,x)))
#im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y4,x)))
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(245, 290)
cbar.ax.set_ylabel('Evaporation temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
  
#error axes
w=0.01 #Error
ax_min = 320
ax_max = 410 #x and y-axes max scale tick
upp_txt = (ax_min+ax_max) / 2.025 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = (ax_min+ax_max) / 1.975 #location of lower error text on plot -- adjust the number to adjust the location
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
plt.ylabel('$T_{dis}$ predicted [K]')
plt.xlabel('$T_{dis}$ measured [K]')
plt.tight_layout()         
plt.savefig('parity_Tdis.pdf')
#plt.show()
plt.close()

  
  
#########################
##### eta_c #######
#########################
#assign axes
y1 = eta_c_corr
y2 = eta_c_corr_dar
y11 = eta_c_ARI
#y3 = my_p_B
#y4 = dom_p
x1 = eta_c_exp
x2 = eta_c_exp_dar
c1 = T_evap
c2 = T_evap_dar
s = 20  # size of points
  
fig, ax = plt.subplots()
im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}\%'.format(mape(y1,x1))+', RMSE = {:0.01f}\%)'.format(rmse(y1,x1)))
im = ax.scatter(x1, y11, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='ARI'+' (MAE = {:0.01f}\%'.format(mape(y11,x1))+', RMSE = {:0.01f}\%)'.format(rmse(y11,x1)))
im = ax.scatter(x2, y2, c=c2, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}\%'.format(mape(y2,x2))+', RMSE = {:0.01f}\%)'.format(rmse(y2,x2)))
#im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y3,x)))
#im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y4,x)))
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(245, 290)
cbar.ax.set_ylabel('Evaporation temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
  
#error axes
w=0.02 #Error
ax_min = 50
ax_max = 80 #x and y-axes max scale tick
upp_txt = (ax_min+ax_max) / 1.85 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = (ax_min+ax_max) / 1.80 #location of lower error text on plot -- adjust the number to adjust the location
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
plt.ylabel('$\\eta_{isen}$ predicted [\%]')
plt.xlabel('$\\eta_{isen}$ measured [\%]')
plt.tight_layout()          
plt.savefig('parity_eta_isen.pdf')
#plt.show()
plt.close()



#########################
##### eta_v #######
#########################
#assign axes
y1 = eta_v_corr
y2 = eta_v_corr_dar
y11 = eta_v_ARI
#y3 = my_p_B
#y4 = dom_p
x1 = eta_v_exp
x2 = eta_v_exp_dar
c1 = T_evap
c2 = T_evap_dar
s = 20  # size of points
  
fig, ax = plt.subplots()
im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}\%'.format(mape(y1,x1))+', RMSE = {:0.01f}\%)'.format(rmse(y1,x1)))
im = ax.scatter(x1, y11, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='ARI'+' (MAE = {:0.01f}\%'.format(mape(y11,x1))+', RMSE = {:0.01f}\%)'.format(rmse(y11,x1)))
im = ax.scatter(x2, y2, c=c2, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}\%'.format(mape(y2,x2))+', RMSE = {:0.01f}\%)'.format(rmse(y2,x2)))
#im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y3,x)))
#im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y4,x)))
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(245, 290)
cbar.ax.set_ylabel('Evaporation temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
  
#error axes
w=0.01 #Error
ax_min = 80
ax_max = 100 #x and y-axes max scale tick
upp_txt = (ax_min+ax_max) / 1.95 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = (ax_min+ax_max) / 1.90 #location of lower error text on plot -- adjust the number to adjust the location
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
plt.ylabel('$\\eta_{v}$ predicted [\%]')
plt.xlabel('$\\eta_{v}$ measured [\%]')
plt.tight_layout()           
plt.savefig('parity_eta_v.pdf')
#plt.show()
plt.close()


#########################
##### f_q #######
#########################
#assign axes
y1 = f_q_corr1
y2 = f_q_corr_dar
y11 = f_q_ARI
#y3 = my_p_B
#y4 = dom_p
x1 = f_q_exp
x2 = f_q_exp_dar
c1 = T_evap
c2 = T_evap_dar
s = 20  # size of points
  
fig, ax = plt.subplots()
im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}\%'.format(mape(y1,x1))+', RMSE = {:0.01f}\%)'.format(rmse(y1,x1)))
im = ax.scatter(x1, y11, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='ARI'+' (MAE = {:0.01f}\%'.format(mape(y11,x1))+', RMSE = {:0.01f}\%)'.format(rmse(y11,x1)))
im = ax.scatter(x2, y2, c=c2, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}\%'.format(mape(y2,x2))+', RMSE = {:0.01f}\%)'.format(rmse(y2,x2)))
#im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y3,x)))
#im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}\%'.format(mape(y4,x)))
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(245, 290)
cbar.ax.set_ylabel('Evaporation temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
  
#error axes
w=0.2 #Error
ax_min = 0
ax_max = 10 #x and y-axes max scale tick
upp_txt = (ax_min+ax_max) / 1.90 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = (ax_min+ax_max) / 1.70 #location of lower error text on plot -- adjust the number to adjust the location
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
plt.ylabel('$f_{loss}$ predicted [\%]')
plt.xlabel('$f_{loss}$ measured [\%]')
plt.tight_layout()           
plt.savefig('parity_floss.pdf')
#plt.show()
plt.close()
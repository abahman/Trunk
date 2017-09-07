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
    golden_mean = (np.sqrt(7)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size
   
pgf_with_latex = {                      # setup matplotlib to use latex for output
"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
"text.usetex": False,                # use LaTeX to write all text
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
    RMSE = np.linalg.norm(predictions - targets) / np.sqrt(n) / np.mean(targets) * 100
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
    
    

condition = ['Total','Two_phase','Vapor']
 
for i in range(len(condition)):
      
    #########################
    ##### m_inj/m_suc #######
    #########################
    #import data from excel file
    df = pd.read_excel('correlation_results.xlsx',sheetname=condition[i]+'_injection_rate_update',header=0) #file name
    #assign axes
    y1 = df['InjectionRatio_pred[i]'][1:]*100
    y2 = df['InjectionRatio_AHRI[i]'][1:]*100
    y3 = df['InjectionRatio_Tello[i]'][1:]*100
    #y11 = df_dar['injectionrate_pred[i]'][1:]*100
    x1 = df['InjectionRatio[i]'][1:]*100
    #x2 = df_dar['injectionrate[i]'][1:]*100
    c1 = (df['T_evap_stpt_ENG[i]'][1:] - 32.0) * 5.0/9.0 + 273.15
    #c2 = df_dar['T_evap[i]'][1:]
    s = 20  # size of points
       
    fig, ax = plt.subplots()
    im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}%'.format(mape(y1,x1))+', RMSE = {:0.01f}%)'.format(rmse(y1,x1)))
    im = ax.scatter(x1, y2, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='AHRI'+' (MAE = {:0.01f}%'.format(mape(y2,x1))+', RMSE = {:0.01f}%)'.format(rmse(y2,x1)))
    im = ax.scatter(x1, y3, c=c1, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Tello'+' (MAE = {:0.01f}%'.format(mape(y3,x1))+', RMSE = {:0.01f}%)'.format(rmse(y3,x1)))
    #im = ax.scatter(x2, y11, c=c2, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}%'.format(mape(y11,x2))+', RMSE = {:0.01f}%)'.format(rmse(y11,x2)))
    #im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}%'.format(mape(y3,x)))
    #im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}%'.format(mape(y4,x)))
    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax)
    # set the color limits
    im.set_clim(245, 290)
    cbar.ax.set_ylabel('Evaporation temperature [K]')
    #ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
       
    #error axes
    error = [0.14,0.02,0.16] #error of ['Total','Two_phase','Vapor']
    w=error[i] #Error
    ax_min = 0
    ax_max = 40 #x and y-axes max scale tick
    upp_txt = (ax_min+ax_max) / 1.8 #location of upper error text on plot -- adjust the number to adjust the location
    low_txt = (ax_min+ax_max) / 1.6 #location of lower error text on plot -- adjust the number to adjust the location
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
    ax.text(low_txt-0.002,low_txt*(1-w),'$-${:0.0f}%'.format(w*100),ha='left',va='top')
    ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}%'.format(w*100),ha='right',va='bottom')
    leg=ax.legend(loc='upper left',scatterpoints=1)
    frame  = leg.get_frame()  
    frame.set_linewidth(0.5)
    ax.set_xlim((ax_min,ax_max))
    ax.set_ylim((ax_min,ax_max))
    plt.ylabel('$\dot m_{inj}$/$\dot m_{suc}$ predicted [%]')
    plt.xlabel('$\dot m_{inj}$/$\dot m_{suc}$ measured [%]')
    plt.tight_layout()       
    plt.savefig(condition[i]+'_parity_m_inj_updated.pdf')
    plt.show()
    plt.close()
    
    #########################
    ##### work (power) ######
    #########################
    #import data from excel file
    df = pd.read_excel('correlation_results.xlsx',sheetname=condition[i]+'_work',header=0) #file name
    #assign axes
    y1 = df['WT1_P_UUT_pred[i]'][1:] #Watts
    y2 = df['WT1_P_UUT_AHRI[i]'][1:] #watss
    y3 = df['WT1_P_UUT_Tello[i]'][1:] #watts
    x1 = df['WT1_P_UUT[i]'][1:] #watts
    c1 = (df['T_evap_stpt_ENG[i]'][1:] - 32.0) * 5.0/9.0 + 273.15
    #c2 = df_dar['T_evap[i]'][1:]
    s = 20  # size of points
         
    fig, ax = plt.subplots()
    im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}%'.format(mape(y1,x1))+', RMSE = {:0.01f}%)'.format(rmse(y1,x1)))
    im = ax.scatter(x1, y2, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='AHRI'+' (MAE = {:0.01f}%'.format(mape(y2,x1))+', RMSE = {:0.01f}%)'.format(rmse(y2,x1)))
    im = ax.scatter(x1, y3, c=c1, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Tello'+' (MAE = {:0.01f}%'.format(mape(y3,x1))+', RMSE = {:0.01f}%)'.format(rmse(y3,x1)))
    #im = ax.scatter(x2, y11, c=c2, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}%'.format(mape(y11,x2))+', RMSE = {:0.01f}%)'.format(rmse(y11,x2)))
    #im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}%'.format(mape(y3,x)))
    #im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}%'.format(mape(y4,x)))
    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax)
    # set the color limits
    im.set_clim(245, 290)
    cbar.ax.set_ylabel('Evaporation temperature [K]')
    #ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
         
    #error axes
    error = [0.05,0.01,0.03] #error of ['Total','Two_phase','Vapor']
    w=error[i] #Error
    ax_min = 2500
    ax_max = 6000 #x and y-axes max scale tick
    upp_txt = (ax_min+ax_max) / 1.9 #location of upper error text on plot -- adjust the number to adjust the location
    low_txt = (ax_min+ax_max) / 1.9 #location of lower error text on plot -- adjust the number to adjust the location
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
    ax.text(low_txt-0.002,low_txt*(1-w),'$-${:0.0f}%'.format(w*100),ha='left',va='top')
    ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}%'.format(w*100),ha='right',va='bottom')
    leg=ax.legend(loc='upper left',scatterpoints=1)
    frame  = leg.get_frame()  
    frame.set_linewidth(0.5)
    ax.set_xlim((ax_min,ax_max))
    ax.set_ylim((ax_min,ax_max))
    plt.ylabel('$\dot W$ predicted [W]')
    plt.xlabel('$\dot W$ measured [W]')
    plt.tight_layout()       
    plt.savefig(condition[i]+'_parity_work.pdf')
    plt.show()
    plt.close()  
       
       
    #########################
    ##### T_dis  #######
    #########################
    #import data from excel file
    df = pd.read_excel('correlation_results.xlsx',sheetname=condition[i]+'_T_dis',header=0) #file name
       
    #assign axes
    y1 = df['T_dis_pred[i]'][1:] #K
    #y2 = df['T_dis_AHRI[i]'][1:] #K
    #y11 = df_dar['injectionrate_pred[i]'][1:]*100
    x1 = df['T_dis[i]'][1:] #K
    #x2 = df_dar['injectionrate[i]'][1:]*100
    c1 = (df['T_evap_stpt_ENG[i]'][1:] - 32.0) * 5.0/9.0 + 273.15
    #c2 = df_dar['T_evap[i]'][1:]
    s = 20  # size of points
          
    fig, ax = plt.subplots()
    im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}%'.format(mape(y1,x1))+', RMSE = {:0.01f}%)'.format(rmse(y1,x1)))
    #im = ax.scatter(x1, y2, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='AHRI'+' (MAE = {:0.01f}%'.format(mape(y2,x1))+', RMSE = {:0.01f}%)'.format(rmse(y2,x1)))
    #im = ax.scatter(x1, y3, c=c1, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Tello'+' (MAE = {:0.01f}%'.format(mape(y3,x1))+', RMSE = {:0.01f}%)'.format(rmse(y3,x1)))
    #im = ax.scatter(x2, y11, c=c2, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}%'.format(mape(y11,x2))+', RMSE = {:0.01f}%)'.format(rmse(y11,x2)))
    #im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}%'.format(mape(y3,x)))
    #im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}%'.format(mape(y4,x)))    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax)
    # set the color limits
    im.set_clim(245, 290)
    cbar.ax.set_ylabel('Evaporation temperature [K]')
    #ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
          
    #error axes
    error = [0.01,0.01,0.02] #error of ['Total','Two_phase','Vapor']
    w=error[i] #Error
    ax_min = 320
    ax_max = 410 #x and y-axes max scale tick
    upp_txt = (ax_min+ax_max) / 2.025 #location of upper error text on plot -- adjust the number to adjust the location
    low_txt = (ax_min+ax_max) / 1.975 #location of lower error text on plot -- adjust the number to adjust the location
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
    ax.text(low_txt-0.002,low_txt*(1-w),'$-${:0.0f}%'.format(w*100),ha='left',va='top')
    ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}%'.format(w*100),ha='right',va='bottom')
    leg=ax.legend(loc='upper left',scatterpoints=1)
    frame  = leg.get_frame()  
    frame.set_linewidth(0.5)
    ax.set_xlim((ax_min,ax_max))
    ax.set_ylim((ax_min,ax_max))
    plt.ylabel('$T_{dis}$ predicted [K]')
    plt.xlabel('$T_{dis}$ measured [K]')
    plt.tight_layout()         
    plt.savefig(condition[i]+'_parity_Tdis.pdf')
    #plt.show()
    plt.close()
   
     
    #########################
    ##### eta_isen #######
    #########################
    #import data from excel file
    df = pd.read_excel('correlation_results.xlsx',sheetname=condition[i]+'_isen_eff',header=0) #file name
   
    #assign axes
    y1 = df['eta_isen_pred[i]'][1:]*100
    #y2 = df['eta_isen_AHRI[i]'][1:]*100
    x1 = df['eta_isen_groll[i]'][1:]
    c1 = (df['T_evap_stpt_ENG[i]'][1:] - 32.0) * 5.0/9.0 + 273.15
    s = 20  # size of points
          
    fig, ax = plt.subplots()
    im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}%'.format(mape(y1,x1))+', RMSE = {:0.01f}%)'.format(rmse(y1,x1)))
    #im = ax.scatter(x1, y2, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='AHRI'+' (MAE = {:0.01f}%'.format(mape(y2,x1))+', RMSE = {:0.01f}%)'.format(rmse(y2,x1)))
    #im = ax.scatter(x1, y3, c=c1, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Tello'+' (MAE = {:0.01f}%'.format(mape(y3,x1))+', RMSE = {:0.01f}%)'.format(rmse(y3,x1)))
    #im = ax.scatter(x2, y11, c=c2, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}%'.format(mape(y11,x2))+', RMSE = {:0.01f}%)'.format(rmse(y11,x2)))
    #im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}%'.format(mape(y3,x)))
    #im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}%'.format(mape(y4,x)))    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax)
    # set the color limits
    im.set_clim(245, 290)
    cbar.ax.set_ylabel('Evaporation temperature [K]')
    #ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
          
    #error axes
    error = [0.02,0.01,0.02] #error of ['Total','Two_phase','Vapor']
    w=error[i] #Error
    ax_min = 50
    ax_max = 80 #x and y-axes max scale tick
    upp_txt = (ax_min+ax_max) / 1.85 #location of upper error text on plot -- adjust the number to adjust the location
    low_txt = (ax_min+ax_max) / 1.80 #location of lower error text on plot -- adjust the number to adjust the location
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
    ax.text(low_txt-0.002,low_txt*(1-w),'$-${:0.0f}%'.format(w*100),ha='left',va='top')
    ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}%'.format(w*100),ha='right',va='bottom')
    leg=ax.legend(loc='upper left',scatterpoints=1)
    frame  = leg.get_frame()  
    frame.set_linewidth(0.5)
    ax.set_xlim((ax_min,ax_max))
    ax.set_ylim((ax_min,ax_max))
    plt.ylabel('$\\eta_{isen}$ predicted [%]')
    plt.xlabel('$\\eta_{isen}$ measured [%]')
    plt.tight_layout()          
    plt.savefig(condition[i]+'_parity_eta_isen.pdf')
    plt.show()
    plt.close()
        
        
        
    #########################
    ##### eta_v #######
    #########################
    #import data from excel file
    df = pd.read_excel('correlation_results.xlsx',sheetname=condition[i]+'_vol_eff',header=0) #file name
   
    #assign axes
    y1 = df['eta_vol_pred[i]'][1:]*100
    #y2 = df['eta_vol_AHRI[i]'][1:]*100
    x1 = df['eta_vol[i]'][1:]*100
    c1 = (df['T_evap_stpt_ENG[i]'][1:] - 32.0) * 5.0/9.0 + 273.15
    s = 20  # size of points
            
    fig, ax = plt.subplots()
    im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}%'.format(mape(y1,x1))+', RMSE = {:0.01f}%)'.format(rmse(y1,x1)))
    #im = ax.scatter(x1, y2, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='AHRI'+' (MAE = {:0.01f}%'.format(mape(y2,x1))+', RMSE = {:0.01f}%)'.format(rmse(y2,x1)))
    #im = ax.scatter(x1, y3, c=c1, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Tello'+' (MAE = {:0.01f}%'.format(mape(y3,x1))+', RMSE = {:0.01f}%)'.format(rmse(y3,x1)))
    #im = ax.scatter(x2, y11, c=c2, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}%'.format(mape(y11,x2))+', RMSE = {:0.01f}%)'.format(rmse(y11,x2)))
    #im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}%'.format(mape(y3,x)))
    #im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}%'.format(mape(y4,x)))    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax)
    # set the color limits
    im.set_clim(245, 290)
    cbar.ax.set_ylabel('Evaporation temperature [K]')
    #ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
            
    #error axes
    error = [0.01,0.01,0.01] #error of ['Total','Two_phase','Vapor']
    w=error[i] #Error
    ax_min = 80
    ax_max = 100 #x and y-axes max scale tick
    upp_txt = (ax_min+ax_max) / 1.95 #location of upper error text on plot -- adjust the number to adjust the location
    low_txt = (ax_min+ax_max) / 1.90 #location of lower error text on plot -- adjust the number to adjust the location
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
    ax.text(low_txt-0.002,low_txt*(1-w),'$-${:0.0f}%'.format(w*100),ha='left',va='top')
    ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}%'.format(w*100),ha='right',va='bottom')
    leg=ax.legend(loc='upper left',scatterpoints=1)
    frame  = leg.get_frame()  
    frame.set_linewidth(0.5)
    ax.set_xlim((ax_min,ax_max))
    ax.set_ylim((ax_min,ax_max))
    plt.ylabel('$\\eta_{v}$ predicted [%]')
    plt.xlabel('$\\eta_{v}$ measured [%]')
    plt.tight_layout()           
    plt.savefig(condition[i]+'_parity_eta_v.pdf')
    plt.show()
    plt.close()
        
        
    #########################
    ##### f_q #######
    #########################
    #import data from excel file
    df = pd.read_excel('correlation_results.xlsx',sheetname=condition[i]+'_heat_loss',header=0) #file name
   
    #assign axes
    y1 = df['f_q_pred[i]'][1:]
    #y2 = df['f_q_AHRI[i]'][1:]
    x1 = df['f_q[i]'][1:]
    c1 = (df['T_evap_stpt_ENG[i]'][1:] - 32.0) * 5.0/9.0 + 273.15
    s = 20  # size of points
          
    fig, ax = plt.subplots()
    im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}%'.format(mape(y1,x1))+', RMSE = {:0.01f}%)'.format(rmse(y1,x1)))
    #im = ax.scatter(x1, y2, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='AHRI'+' (MAE = {:0.01f}%'.format(mape(y2,x1))+', RMSE = {:0.01f}%)'.format(rmse(y2,x1)))
    #im = ax.scatter(x1, y3, c=c1, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Tello'+' (MAE = {:0.01f}%'.format(mape(y3,x1))+', RMSE = {:0.01f}%)'.format(rmse(y3,x1)))
    #im = ax.scatter(x2, y11, c=c2, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}%'.format(mape(y11,x2))+', RMSE = {:0.01f}%)'.format(rmse(y11,x2)))
    #im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}%'.format(mape(y3,x)))
    #im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}%'.format(mape(y4,x)))    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax)
    # set the color limits
    im.set_clim(245, 290)
    cbar.ax.set_ylabel('Evaporation temperature [K]')
    #ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
          
    #error axes
    error = [0.12,0.10,0.05] #error of ['Total','Two_phase','Vapor']
    w=error[i] #Error
    ax_min = 0
    ax_max = 10 #x and y-axes max scale tick
    upp_txt = (ax_min+ax_max) / 1.85 #location of upper error text on plot -- adjust the number to adjust the location
    low_txt = (ax_min+ax_max) / 1.70 #location of lower error text on plot -- adjust the number to adjust the location
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
    ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
    ax.text(low_txt-0.002,low_txt*(1-w),'$-${:0.0f}%'.format(w*100),ha='left',va='top')
    ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}%'.format(w*100),ha='right',va='bottom')
    leg=ax.legend(loc='upper left',scatterpoints=1)
    frame  = leg.get_frame()  
    frame.set_linewidth(0.5)
    ax.set_xlim((ax_min,ax_max))
    ax.set_ylim((ax_min,ax_max))
    plt.ylabel('$f_{q}$ predicted [%]')
    plt.xlabel('$f_{q}$ measured [%]')
    plt.tight_layout()           
    plt.savefig(condition[i]+'_parity_f_q.pdf')
    plt.show()
    plt.close()



    

####Dardenne plots##### 


#########################
##### m_inj/m_suc #######
#########################
#import data from excel file
df = pd.read_excel('correlation_results.xlsx',sheetname='Dardenne_injection_rate_update',header=0) #file name
#assign axes
y1 = df['InjectionRatio_pred[i]'][1:]*100
y2 = df['InjectionRatio_AHRI[i]'][1:]*100
y3 = df['InjectionRatio_Tello[i]'][1:]*100
#y11 = df_dar['injectionrate_pred[i]'][1:]*100
x1 = df['InjectionRatio[i]'][1:]*100
#x2 = df_dar['injectionrate[i]'][1:]*100
c1 = (df['T_evap_stpt_ENG[i]'][1:] - 32.0) * 5.0/9.0 + 273.15
#c2 = df_dar['T_evap[i]'][1:]
s = 20  # size of points
   
fig, ax = plt.subplots()
im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}%'.format(mape(y1,x1))+', RMSE = {:0.01f}%)'.format(rmse(y1,x1)))
im = ax.scatter(x1, y2, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='AHRI'+' (MAE = {:0.01f}%'.format(mape(y2,x1))+', RMSE = {:0.01f}%)'.format(rmse(y2,x1)))
im = ax.scatter(x1, y3, c=c1, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Tello'+' (MAE = {:0.01f}%'.format(mape(y3,x1))+', RMSE = {:0.01f}%)'.format(rmse(y3,x1)))
#im = ax.scatter(x2, y11, c=c2, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}%'.format(mape(y11,x2))+', RMSE = {:0.01f}%)'.format(rmse(y11,x2)))
#im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}%'.format(mape(y3,x)))
#im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}%'.format(mape(y4,x)))
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(245, 290)
cbar.ax.set_ylabel('Evaporation temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
   
#error axes
w=0.12 #Error
ax_min = 0
ax_max = 80 #x and y-axes max scale tick
upp_txt = (ax_min+ax_max) / 1.5 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = (ax_min+ax_max) / 1.25 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'$-${:0.0f}%'.format(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}%'.format(w*100),ha='right',va='bottom')
leg=ax.legend(loc='upper left',scatterpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
ax.set_xlim((ax_min,ax_max))
ax.set_ylim((ax_min,ax_max))
plt.ylabel('$\dot m_{inj}$/$\dot m_{suc}$ predicted [%]')
plt.xlabel('$\dot m_{inj}$/$\dot m_{suc}$ measured [%]')
plt.tight_layout()       
plt.savefig('Dardenne_parity_m_inj_updated.pdf')
plt.show()
plt.close()
  
#########################
##### work (power) ######
#########################
#import data from excel file
df = pd.read_excel('correlation_results.xlsx',sheetname='Dardenne_work',header=0) #file name
#assign axes
y1 = df['W_dot_pred[i]'][1:] #Watts
y2 = df['W_dot_AHRI[i]'][1:] #watss
y3 = df['W_dot_Tello[i]'][1:] #watts
x1 = df['W_dot[i]'][1:] #watts
c1 = df['T_evap[i]'][1:] #K
#c2 = df_dar['T_evap[i]'][1:]
s = 20  # size of points
     
fig, ax = plt.subplots()
im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}%'.format(mape(y1,x1))+', RMSE = {:0.01f}%)'.format(rmse(y1,x1)))
im = ax.scatter(x1, y2, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='AHRI'+' (MAE = {:0.01f}%'.format(mape(y2,x1))+', RMSE = {:0.01f}%)'.format(rmse(y2,x1)))
im = ax.scatter(x1, y3, c=c1, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Tello'+' (MAE = {:0.01f}%'.format(mape(y3,x1))+', RMSE = {:0.01f}%)'.format(rmse(y3,x1)))
#im = ax.scatter(x2, y11, c=c2, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}%'.format(mape(y11,x2))+', RMSE = {:0.01f}%)'.format(rmse(y11,x2)))
#im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}%'.format(mape(y3,x)))
#im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}%'.format(mape(y4,x)))
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(245, 290)
cbar.ax.set_ylabel('Evaporation temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
     
#error axes
w=0.3 #Error
ax_min = 0
ax_max = 8000 #x and y-axes max scale tick
upp_txt = (ax_min+ax_max) / 1.8 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = (ax_min+ax_max) / 1.3 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'$-${:0.0f}%'.format(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}%'.format(w*100),ha='right',va='bottom')
leg=ax.legend(loc='upper left',scatterpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
ax.set_xlim((ax_min,ax_max))
ax.set_ylim((ax_min,ax_max))
plt.ylabel('$\dot W$ predicted [W]')
plt.xlabel('$\dot W$ measured [W]')
plt.tight_layout()       
plt.savefig('Dardenne_parity_work.pdf')
plt.show()
plt.close()  
   
   
#########################
##### T_dis  #######
#########################
#import data from excel file
df = pd.read_excel('correlation_results.xlsx',sheetname='Dardenne_T_dis',header=0) #file name
   
#assign axes
y1 = df['T_dis_pred[i]'][1:] #K
#y2 = df['T_dis_AHRI[i]'][1:] #K
x1 = df['T_dis[i]'][1:] #K
c1 = df['T_evap[i]'][1:] #K
s = 20  # size of points
      
fig, ax = plt.subplots()
im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}%'.format(mape(y1,x1))+', RMSE = {:0.01f}%)'.format(rmse(y1,x1)))
#im = ax.scatter(x1, y2, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='AHRI'+' (MAE = {:0.01f}%'.format(mape(y2,x1))+', RMSE = {:0.01f}%)'.format(rmse(y2,x1)))
#im = ax.scatter(x1, y3, c=c1, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Tello'+' (MAE = {:0.01f}%'.format(mape(y3,x1))+', RMSE = {:0.01f}%)'.format(rmse(y3,x1)))
#im = ax.scatter(x2, y11, c=c2, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}%'.format(mape(y11,x2))+', RMSE = {:0.01f}%)'.format(rmse(y11,x2)))
#im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}%'.format(mape(y3,x)))
#im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}%'.format(mape(y4,x)))    # Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(245, 290)
cbar.ax.set_ylabel('Evaporation temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
      
#error axes
w = 0.01 #error
ax_min = 320
ax_max = 410 #x and y-axes max scale tick
upp_txt = (ax_min+ax_max) / 2.025 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = (ax_min+ax_max) / 1.975 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'$-${:0.0f}%'.format(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}%'.format(w*100),ha='right',va='bottom')
leg=ax.legend(loc='upper left',scatterpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
ax.set_xlim((ax_min,ax_max))
ax.set_ylim((ax_min,ax_max))
plt.ylabel('$T_{dis}$ predicted [K]')
plt.xlabel('$T_{dis}$ measured [K]')
plt.tight_layout()         
plt.savefig('Dardenne_parity_Tdis.pdf')
plt.show()
plt.close()
   
   
#########################
##### eta_isen #######
#########################
#import data from excel file
df = pd.read_excel('correlation_results.xlsx',sheetname='Dardenne_isen_eff',header=0) #file name
   
#assign axes
y1 = df['eta_isen_pred[i]'][1:]*100
#y2 = df['eta_isen_AHRI[i]'][1:]*100
x1 = df['eta_isen_groll[i]'][1:]*100
c1 = df['T_evap[i]'][1:]
s = 20  # size of points
      
fig, ax = plt.subplots()
im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}%'.format(mape(y1,x1))+', RMSE = {:0.01f}%)'.format(rmse(y1,x1)))
#im = ax.scatter(x1, y2, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='AHRI'+' (MAE = {:0.01f}%'.format(mape(y2,x1))+', RMSE = {:0.01f}%)'.format(rmse(y2,x1)))
#im = ax.scatter(x1, y3, c=c1, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Tello'+' (MAE = {:0.01f}%'.format(mape(y3,x1))+', RMSE = {:0.01f}%)'.format(rmse(y3,x1)))
#im = ax.scatter(x2, y11, c=c2, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}%'.format(mape(y11,x2))+', RMSE = {:0.01f}%)'.format(rmse(y11,x2)))
#im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}%'.format(mape(y3,x)))
#im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}%'.format(mape(y4,x)))    # Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(245, 290)
cbar.ax.set_ylabel('Evaporation temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
      
#error axes
w=0.05 #Error
ax_min = 50
ax_max = 80 #x and y-axes max scale tick
upp_txt = (ax_min+ax_max) / 1.85 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = (ax_min+ax_max) / 1.80 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'$-${:0.0f}%'.format(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}%'.format(w*100),ha='right',va='bottom')
leg=ax.legend(loc='upper left',scatterpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
ax.set_xlim((ax_min,ax_max))
ax.set_ylim((ax_min,ax_max))
plt.ylabel('$\\eta_{isen}$ predicted [%]')
plt.xlabel('$\\eta_{isen}$ measured [%]')
plt.tight_layout()          
plt.savefig('Dardenne_parity_eta_isen.pdf')
plt.show()
plt.close()
    
    
    
#########################
##### eta_v #######
#########################
#import data from excel file
df = pd.read_excel('correlation_results.xlsx',sheetname='Dardenne_vol_eff',header=0) #file name
   
#assign axes
y1 = df['eta_vol_pred[i]'][1:]*100
#y2 = df['eta_vol_AHRI[i]'][1:]*100
x1 = df['eta_vol[i]'][1:]*100
c1 = df['T_evap[i]'][1:]
s = 20  # size of points
        
fig, ax = plt.subplots()
im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}%'.format(mape(y1,x1))+', RMSE = {:0.01f}%)'.format(rmse(y1,x1)))
#im = ax.scatter(x1, y2, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='AHRI'+' (MAE = {:0.01f}%'.format(mape(y2,x1))+', RMSE = {:0.01f}%)'.format(rmse(y2,x1)))
#im = ax.scatter(x1, y3, c=c1, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Tello'+' (MAE = {:0.01f}%'.format(mape(y3,x1))+', RMSE = {:0.01f}%)'.format(rmse(y3,x1)))
#im = ax.scatter(x2, y11, c=c2, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}%'.format(mape(y11,x2))+', RMSE = {:0.01f}%)'.format(rmse(y11,x2)))
#im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}%'.format(mape(y3,x)))
#im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}%'.format(mape(y4,x)))    # Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(245, 290)
cbar.ax.set_ylabel('Evaporation temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
        
#error axes
w=0.03 #Error
ax_min = 80
ax_max = 100 #x and y-axes max scale tick
upp_txt = (ax_min+ax_max) / 1.95 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = (ax_min+ax_max) / 1.9 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'$-${:0.0f}%'.format(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}%'.format(w*100),ha='right',va='bottom')
leg=ax.legend(loc='upper left',scatterpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
ax.set_xlim((ax_min,ax_max))
ax.set_ylim((ax_min,ax_max))
plt.ylabel('$\\eta_{v}$ predicted [%]')
plt.xlabel('$\\eta_{v}$ measured [%]')
plt.tight_layout()           
plt.savefig('Dardenne_parity_eta_v.pdf')
plt.show()
plt.close()
    
    
#########################
##### f_q #######
#########################
#import data from excel file
df = pd.read_excel('correlation_results.xlsx',sheetname='Dardenne_heat_loss',header=0) #file name
   
#assign axes
y1 = df['f_q_pred[i]'][1:]
#y2 = df['f_q_AHRI[i]'][1:]
x1 = df['f_q[i]'][1:]
c1 = df['T_evap[i]'][1:]
s = 20  # size of points
      
fig, ax = plt.subplots()
im = ax.scatter(x1, y1, c=c1, s=s, cmap=plt.cm.jet, marker='^',lw=0.2, alpha =0.9,label='Dimensionless $\Pi$'+' (MAE = {:0.01f}%'.format(mape(y1,x1))+', RMSE = {:0.01f}%)'.format(rmse(y1,x1)))
#im = ax.scatter(x1, y2, c=c1, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, alpha =0.9,label='AHRI'+' (MAE = {:0.01f}%'.format(mape(y2,x1))+', RMSE = {:0.01f}%)'.format(rmse(y2,x1)))
#im = ax.scatter(x1, y3, c=c1, s=s, cmap=plt.cm.jet, marker='s',lw=0.2, alpha =0.9,label='Tello'+' (MAE = {:0.01f}%'.format(mape(y3,x1))+', RMSE = {:0.01f}%)'.format(rmse(y3,x1)))
#im = ax.scatter(x2, y11, c=c2, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, alpha =0.9,label='Dardenne'+' (MAE = {:0.01f}%'.format(mape(y11,x2))+', RMSE = {:0.01f}%)'.format(rmse(y11,x2)))
#im = ax.scatter(x, y3, c=c, s=s, cmap=plt.cm.jet, marker='d',lw=0.2, label='$\\pi = f \\left( \\frac{p_{dis}}{p_{suc}},  \\frac{p_{inj}}{p_{suc}}, \\frac{\\Delta h_{inj}}{\\Delta h_{fg,inj}},\\frac{\\Delta h_{suc}}{\\Delta h_{fg,suc}} \\right)$'+' MAE = {:0.1f}%'.format(mape(y3,x)))
#im = ax.scatter(x, y4, c=c, s=s, cmap=plt.cm.jet, marker='o',lw=0.2, label='$\\pi = f \\left( T_{evap}, T_{cond}, T_{dew,inj} \\right)$'+' MAE = {:0.1f}%'.format(mape(y4,x)))    # Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(245, 290)
cbar.ax.set_ylabel('Evaporation temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
      
#error axes
w=0.1 #Error
ax_min = 0
ax_max = 10 #x and y-axes max scale tick
upp_txt = (ax_min+ax_max) / 1.90 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = (ax_min+ax_max) / 1.70 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'$-${:0.0f}%'.format(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+{:0.0f}%'.format(w*100),ha='right',va='bottom')
leg=ax.legend(loc='upper left',scatterpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
ax.set_xlim((ax_min,ax_max))
ax.set_ylim((ax_min,ax_max))
plt.ylabel('$f_{q}$ predicted [%]')
plt.xlabel('$f_{q}$ measured [%]')
plt.tight_layout()           
plt.savefig('Dardenne_parity_f_q.pdf')
plt.show()
plt.close()
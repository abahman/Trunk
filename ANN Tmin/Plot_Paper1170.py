#Here we import some python packages
from __future__ import division
from math import pi, cos, sin, asin,tan,sqrt
from scipy.constants import g
from scipy.integrate import quad,simps
from time import clock
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import pylab
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import os, sys
plt.style.use('Elsevier.mplstyle')

#Imports from CoolProp (fluid property database)
from CoolProp import State,AbstractState
from CoolProp import CoolProp as CP

  
    
def savefigs(name):
    plt.show()
    #plt.savefig(name+'.eps')
    #plt.savefig(name+'.pdf')
    #plt.savefig(name+'.png',dpi=600)
    plt.close()


def MAPE(y_true,y_pred):
    """
    Function returning the Mean Absolute Percentage Error
    """

    return np.mean(np.abs((y_true - y_pred) / y_true))*100 
    
def REmean(y_true,y_pred):
    
    return np.mean(np.fabs(y_true - y_pred)/y_true)

def RE(y_true,y_pred):
    """
    Function returning the Maximum Relative Percentage Error
    """

    return np.max(np.abs((y_true - y_pred) / y_true))*100 
    
def ParityPlots_ScrollExpander(df_calc_ANN,df_calc_SemiEmp,df_meas,**kwargs):
    
    # Measured data
    mdot_meas = np.asarray(df_meas['mdot'].values.T[:].tolist()) #[kg/s]
    Tdis_meas = np.asarray(df_meas['Tdis'].values.T[:].tolist()) #[K]
    Wdot_meas = np.asarray(df_meas['Wdot'].values.T[:].tolist()) #[W]
    etais_meas = np.asarray(df_meas['etais'].values.T[:].tolist()) #[-] 

    # Calculated data ANN
    mdot_calc_ANN = np.asarray(df_calc_ANN['mdot'].values.T[:].tolist()) #[kg/s]
    Tdis_calc_ANN = np.asarray(df_calc_ANN['Tdis'].values.T[:].tolist()) #[K]
    Wdot_calc_ANN = np.asarray(df_calc_ANN['Wdot'].values.T[:].tolist()) #[W] 
    etais_calc_ANN = np.asarray(df_calc_ANN['etais'].values.T[:].tolist()) #[-]   

    # Measured data semi-empirical model
    mdot_calc_SemiEmp = np.asarray(df_calc_SemiEmp['mdot'].values.T[:].tolist()) #[kg/s]
    Tdis_calc_SemiEmp = np.asarray(df_calc_SemiEmp['Tdis'].values.T[:].tolist()) #[K]
    Wdot_calc_SemiEmp = np.asarray(df_calc_SemiEmp['Wdot'].values.T[:].tolist()) #[W] 
    etais_calc_SemiEmp = np.asarray(df_calc_SemiEmp['etais'].values.T[:].tolist()) #[-]   

    ## Calculate MAEs
    MAPE_Wdot_ANN = MAPE(np.asarray(Wdot_meas),np.asarray(Wdot_calc_ANN))
    MAPE_mdot_ANN = MAPE(np.asarray(mdot_meas),np.asarray(mdot_calc_ANN))
    MAPE_Tdis_ANN = MAPE(np.asarray(Tdis_meas),np.asarray(Tdis_calc_ANN))
    MAPE_etais_ANN = MAPE(np.asarray(etais_meas),np.asarray(etais_calc_ANN))
    
    MAPE_Wdot_SemiEmp = MAPE(np.asarray(Wdot_meas),np.asarray(Wdot_calc_SemiEmp))
    MAPE_mdot_SemiEmp = MAPE(np.asarray(mdot_meas),np.asarray(mdot_calc_SemiEmp))
    MAPE_Tdis_SemiEmp = MAPE(np.asarray(Tdis_meas),np.asarray(Tdis_calc_SemiEmp))
    MAPE_etais_SemiEmp = MAPE(np.asarray(etais_meas),np.asarray(etais_calc_SemiEmp))
    

    ## Plot Tdis
    plt.figure(figsize=(3.5,2.5))

    plt.plot(Tdis_calc_SemiEmp,Tdis_meas,'o',ms = 4,mfc = 'none',mec='b', label='Semi-emp (MAPE = {:0.1f}%)'.format(MAPE_Tdis_SemiEmp))    
    plt.plot(Tdis_calc_ANN,Tdis_meas,'d',ms = 5,mfc = 'none',mec='g', label='ANN (MAPE = {:0.1f}%)'.format(MAPE_Tdis_ANN))

    plt.xlabel('Discharge Temperature (calc) [K]')
    plt.ylabel('Discharge Temperature\n(meas) [K]')
    
    Tmin = 310
    Tmax = 370
    x=[Tmin,Tmax]
    y=[Tmin,Tmax]
    y105=[3+Tmin,3+Tmax]
    y95=[Tmin-3,Tmax-3]
    
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)
    
    plt.xlim(Tmin,Tmax)
    plt.ylim(Tmin,Tmax)
    plt.legend(loc='upper left',fontsize = 9)
    plt.tight_layout(pad=0.2)
    savefigs('Td_OilFree_Scroll_R245fa')


    ## Validation mass flow rate
    plt.figure(figsize=(3.5,2.5))
    
    plt.plot(mdot_calc_SemiEmp,mdot_meas,'o',ms = 4,mfc = 'none',mec='b', label='Semi-emp (MAPE = {:0.1f}%)'.format(MAPE_mdot_SemiEmp))
    plt.plot(mdot_calc_ANN,mdot_meas,'d',ms = 5,mfc = 'none',mec='g', label='ANN (MAPE = {:0.1f}%)'.format(MAPE_mdot_ANN))

    plt.xlabel('Mass flow rate (calc) [kg/s]')
    plt.ylabel('Mass flow rate (meas) [kg/s]')
    
    Mmin = 0.05
    Mmax = 0.25
    x=[Mmin,Mmax]
    y=[Mmin,Mmax]
    y105=[1.05*Mmin,1.05*Mmax]
    y95=[0.95*Mmin,0.95*Mmax]
    
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)
    
    plt.xlim(Mmin,Mmax)
    plt.ylim(Mmin,Mmax)
    plt.legend(loc='upper left',fontsize = 9)
    plt.tight_layout(pad=0.2)
    savefigs('mdot_OilFree_Scroll_R245fa')
    
    
    ## Validation power output
    plt.figure(figsize=(3.5,2.5))
    
    plt.plot(Wdot_calc_SemiEmp/1000,Wdot_meas/1000,'o',ms = 4,mfc = 'none',mec='b', label='Semi-emp (MAPE = {:0.1f}%)'.format(MAPE_Wdot_SemiEmp))
    plt.plot(Wdot_calc_ANN/1000,Wdot_meas/1000,'d',ms = 5,mfc = 'none',mec='g', label='ANN (MAPE = {:0.1f}%)'.format(MAPE_Wdot_ANN))

    plt.ylabel('Shaft Power (calc) [kW]')
    plt.xlabel('Shaft Power (meas) [kW]')
    
    Wmin = 0
    Wmax = 4000
    x=[Wmin/1000,Wmax/1000]
    y=[Wmin/1000,Wmax/1000]
    y105=[1.1*Wmin/1000,1.1*Wmax/1000]
    y95=[0.9*Wmin/1000,0.9*Wmax/1000]
    
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)
    plt.xlim(Wmin/1000,Wmax/1000)
    plt.ylim(Wmin/1000,Wmax/1000)
    plt.legend(loc='upper left',fontsize = 9)    
    plt.tight_layout(pad=0.2)
    savefigs('Wdot_OilFree_Scroll_R245fa')

    ## Validation isentropic efficiency
    plt.figure(figsize=(3.5,2.5))
    
    plt.plot(etais_calc_SemiEmp,etais_meas,'o',ms = 4,mfc = 'none',mec='b', label='Semi-emp (MAPE = {:0.1f}%)'.format(MAPE_etais_SemiEmp))
    plt.plot(etais_calc_ANN,etais_meas,'d',ms = 5,mfc = 'none',mec='g', label='ANN (MAPE = {:0.1f}%)'.format(MAPE_etais_ANN))
    
    plt.ylabel('Isentropic Efficiency (calc) [-]')
    plt.xlabel('Isentropic Efficiency (meas) [-]')
    
    eta_min = 0.1
    eta_max = 0.7
    x=[eta_min,eta_max]
    y=[eta_min,eta_max]
    y105=[1.1*eta_min,1.1*eta_max]
    y95=[0.9*eta_min,0.9*eta_max]
    
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)
    
    plt.xlim(0.1,0.7)
    plt.ylim(0.1,0.7)
    plt.legend(loc='upper left',fontsize = 9)    
    plt.tight_layout(pad=0.2)
    savefigs('etas_OilFree_Scroll_R245fa')


def ParityPlots_VIScrollComp(df_calc_ANN,df_calc_ANN_2phase,df_calc_BPI,df_calc_BPI_2phase,df_meas_comp_SH,df_meas_comp_2phase):
    
    # Measured data
    mdot_meas_SH = np.asarray(df_meas_comp_SH['mdot'].values.T[:].tolist())/1000 #[kg/s]
    mdotinj_meas_SH = np.asarray(df_meas_comp_SH['mdot_inj'].values.T[:].tolist())/1000 #[kg/s]
    Tdis_meas_SH = np.asarray(df_meas_comp_SH['Tdis'].values.T[:].tolist()) #[K]
    Wdot_meas_SH = np.asarray(df_meas_comp_SH['Wdot'].values.T[:].tolist()) #[W]
    etais_meas_SH = np.asarray(df_meas_comp_SH['etaoa'].values.T[:].tolist()) #[-] 
    fq_meas_SH = np.asarray(df_meas_comp_SH['fq'].values.T[:].tolist()) #[-]

    mdot_meas_2phase = np.asarray(df_meas_comp_2phase['mdot'].values.T[:].tolist()) #[kg/s]
    mdotinj_meas_2phase = np.asarray(df_meas_comp_2phase['mdot_inj'].values.T[:].tolist()) #[kg/s]
    Tdis_meas_2phase = np.asarray(df_meas_comp_2phase['Tdis'].values.T[:].tolist()) #[K]
    Wdot_meas_2phase = np.asarray(df_meas_comp_2phase['Wdot'].values.T[:].tolist()) #[W]
    etais_meas_2phase = np.asarray(df_meas_comp_2phase['etaoa'].values.T[:].tolist()) #[-] 
    fq_meas_2phase = np.asarray(df_meas_comp_2phase['fq'].values.T[:].tolist()) #[-]
    
    # Calculated data ANN
    mdot_calc_ANN = np.asarray(df_calc_ANN['mdot'].values.T[:].tolist()) #[kg/s]
    mdotinj_calc_ANN = np.asarray(df_calc_ANN['mdot_inj'].values.T[:].tolist()) #[kg/s]
    Tdis_calc_ANN = np.asarray(df_calc_ANN['Tdis'].values.T[:].tolist()) #[K]
    Wdot_calc_ANN = np.asarray(df_calc_ANN['Wdot'].values.T[:].tolist()) #[W] 
    etais_calc_ANN = np.asarray(df_calc_ANN['etaoa'].values.T[:].tolist()) #[-]   
    fq_calc_ANN = np.asarray(df_calc_ANN['fq'].values.T[:].tolist()) #[-]   

    # Calculated data ANN 2-phase
    mdot_calc_ANN_2phase = np.asarray(df_calc_ANN_2phase['mdot'].values.T[:].tolist()) #[kg/s]
    mdotinj_calc_ANN_2phase = np.asarray(df_calc_ANN_2phase['mdot_inj'].values.T[:].tolist()) #[kg/s]
    Tdis_calc_ANN_2phase = np.asarray(df_calc_ANN_2phase['Tdis'].values.T[:].tolist()) #[K]
    Wdot_calc_ANN_2phase = np.asarray(df_calc_ANN_2phase['Wdot'].values.T[:].tolist()) #[W] 
    etais_calc_ANN_2phase = np.asarray(df_calc_ANN_2phase['etaoa'].values.T[:].tolist()) #[-] 
    fq_calc_ANN_2phase = np.asarray(df_calc_ANN_2phase['fq'].values.T[:].tolist()) #[-]

    # Calculated data B-PI
    mdot_calc_BPI = np.asarray(df_calc_BPI['mdot'].values.T[:].tolist()) #[kg/s]
    mdotinj_calc_BPI = np.asarray(df_calc_BPI['mdot_inj'].values.T[:].tolist()) #[kg/s]
    Tdis_calc_BPI = np.asarray(df_calc_BPI['Tdis'].values.T[:].tolist()) #[K]
    Wdot_calc_BPI = np.asarray(df_calc_BPI['Wdot'].values.T[:].tolist()) #[W] 
    etais_calc_BPI = np.asarray(df_calc_BPI['etaoa'].values.T[:].tolist()) #[-]   
    fq_calc_BPI = np.asarray(df_calc_BPI['fq'].values.T[:].tolist()) #[-]   

    # Calculated data B-PI 2-phase
    mdot_calc_BPI_2phase = np.asarray(df_calc_BPI_2phase['mdot'].values.T[:].tolist()) #[kg/s]
    mdotinj_calc_BPI_2phase = np.asarray(df_calc_BPI_2phase['mdot_inj'].values.T[:].tolist()) #[kg/s]
    Tdis_calc_BPI_2phase = np.asarray(df_calc_BPI_2phase['Tdis'].values.T[:].tolist()) #[K]
    Wdot_calc_BPI_2phase = np.asarray(df_calc_BPI_2phase['Wdot'].values.T[:].tolist()) #[W] 
    etais_calc_BPI_2phase = np.asarray(df_calc_BPI_2phase['etaoa'].values.T[:].tolist()) #[-] 
    fq_calc_BPI_2phase = np.asarray(df_calc_BPI_2phase['fq'].values.T[:].tolist()) #[-]  

    ## Calculate MAEs
    MAPE_Wdot_ANN = MAPE(np.asarray(Wdot_meas_SH),np.asarray(Wdot_calc_ANN))
    MAPE_mdot_ANN = MAPE(np.asarray(mdot_meas_SH),np.asarray(mdot_calc_ANN))
    MAPE_mdotinj_ANN = MAPE(np.asarray(mdotinj_meas_SH),np.asarray(mdotinj_calc_ANN))
    MAPE_Tdis_ANN = MAPE(np.asarray(Tdis_meas_SH),np.asarray(Tdis_calc_ANN))
    MAPE_etais_ANN = MAPE(np.asarray(etais_meas_SH),np.asarray(etais_calc_ANN))
    MAPE_fq_ANN = MAPE(np.asarray(fq_meas_SH),np.asarray(fq_calc_ANN))

    MAPE_Wdot_ANN_2phase = MAPE(np.asarray(Wdot_meas_2phase),np.asarray(Wdot_calc_ANN_2phase))
    MAPE_mdot_ANN_2phase = MAPE(np.asarray(mdot_meas_2phase),np.asarray(mdot_calc_ANN_2phase))
    MAPE_mdotinj_ANN_2phase = MAPE(np.asarray(mdotinj_meas_2phase),np.asarray(mdotinj_calc_ANN_2phase))
    MAPE_Tdis_ANN_2phase = MAPE(np.asarray(Tdis_meas_2phase),np.asarray(Tdis_calc_ANN_2phase))
    MAPE_etais_ANN_2phase = MAPE(np.asarray(etais_meas_2phase),np.asarray(etais_calc_ANN_2phase))
    MAPE_fq_ANN_2phase = MAPE(np.asarray(fq_meas_2phase),np.asarray(fq_calc_ANN_2phase))

    MAPE_Wdot_BPI = MAPE(np.asarray(Wdot_meas_SH),np.asarray(Wdot_calc_BPI))
    MAPE_mdot_BPI = MAPE(np.asarray(mdot_meas_SH),np.asarray(mdot_calc_BPI))
    MAPE_mdotinj_BPI = MAPE(np.asarray(mdotinj_meas_SH),np.asarray(mdotinj_calc_BPI))
    MAPE_Tdis_BPI = MAPE(np.asarray(Tdis_meas_SH),np.asarray(Tdis_calc_BPI))
    MAPE_etais_BPI = MAPE(np.asarray(etais_meas_SH),np.asarray(etais_calc_BPI))
    MAPE_fq_BPI = MAPE(np.asarray(fq_meas_SH),np.asarray(fq_calc_BPI))

    MAPE_Wdot_BPI_2phase = MAPE(np.asarray(Wdot_meas_2phase),np.asarray(Wdot_calc_BPI_2phase))
    MAPE_mdot_BPI_2phase = MAPE(np.asarray(mdot_meas_2phase),np.asarray(mdot_calc_BPI_2phase))
    MAPE_mdotinj_BPI_2phase = MAPE(np.asarray(mdotinj_meas_2phase),np.asarray(mdotinj_calc_BPI_2phase))
    MAPE_Tdis_BPI_2phase = MAPE(np.asarray(Tdis_meas_2phase),np.asarray(Tdis_calc_BPI_2phase))
    MAPE_etais_BPI_2phase = MAPE(np.asarray(etais_meas_2phase),np.asarray(etais_calc_BPI_2phase))
    MAPE_fq_BPI_2phase = MAPE(np.asarray(fq_meas_2phase),np.asarray(fq_calc_BPI_2phase))    


    ## Plot Tdis
    plt.figure(figsize=(3.5,2.5))

    plt.plot(Tdis_calc_ANN,Tdis_meas_SH,'o',ms = 4,mfc = 'none',mec='b', label='ANN (MAPE = {:0.1f}%)'.format(MAPE_Tdis_ANN))    
    plt.plot(Tdis_calc_ANN_2phase,Tdis_meas_2phase,'d',ms = 5,mfc = 'none',mec='g', label='ANN 2$\phi$ (MAPE = {:0.1f}%)'.format(MAPE_Tdis_ANN_2phase))
    plt.plot(Tdis_calc_BPI,Tdis_meas_SH,'s',ms = 4,mfc = 'none',mec='r', label='B-$\pi$ (MAPE = {:0.1f}%)'.format(MAPE_Tdis_BPI))    
    plt.plot(Tdis_calc_BPI_2phase,Tdis_meas_2phase,'^',ms = 5,mfc = 'none',mec='y', label='B-$\pi$ 2$\phi$ (MAPE = {:0.1f}%)'.format(MAPE_Tdis_BPI_2phase))

    plt.xlabel('Discharge Temperature (calc) [K]')
    plt.ylabel('Discharge Temperature\n(meas) [K]')
    
    Tmin = 340
    Tmax = 380
    x=[Tmin,Tmax]
    y=[Tmin,Tmax]
    y105=[3+Tmin,3+Tmax]
    y95=[Tmin-3,Tmax-3]
    
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)
    
    plt.text(375,365,'$\pm$3K',fontsize=9)
    
    plt.xlim(Tmin,Tmax)
    plt.ylim(Tmin,Tmax)
    plt.legend(loc='upper left',fontsize = 8)
    plt.tight_layout(pad=0.2)
    savefigs('Td_VIScroll_R407C')


    ## Validation mass flow rate
    plt.figure(figsize=(3.5,2.5))
    
    plt.plot(mdot_calc_ANN,mdot_meas_SH,'o',ms = 4,mfc = 'none',mec='b', label='ANN (MAPE = {:0.1f}%)'.format(MAPE_mdot_ANN))
    plt.plot(mdot_calc_ANN_2phase,mdot_meas_2phase,'d',ms = 5,mfc = 'none',mec='g', label='ANN 2$\phi$ (MAPE = {:0.1f}%)'.format(MAPE_mdot_ANN_2phase))
    plt.plot(mdot_calc_BPI,mdot_meas_SH,'s',ms = 4,mfc = 'none',mec='r', label='B-$\pi$ (MAPE = {:0.1f}%)'.format(MAPE_mdot_BPI))
    plt.plot(mdot_calc_BPI_2phase,mdot_meas_2phase,'^',ms = 5,mfc = 'none',mec='y', label='B-$\pi$ 2$\phi$ (MAPE = {:0.1f}%)'.format(MAPE_mdot_BPI_2phase))
    
    plt.xlabel('Mass flow rate (calc) [kg/s]')
    plt.ylabel('Mass flow rate (meas) [kg/s]')
    
    Mmin = 0.04
    Mmax = 0.1
    x=[Mmin,Mmax]
    y=[Mmin,Mmax]
    y105=[1.05*Mmin,1.05*Mmax]
    y95=[0.95*Mmin,0.95*Mmax]
    
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)
    
    plt.text(0.09,0.075,'$\pm$5%',fontsize=9)
    
    plt.plot(loc=2,fontsize=8)
    plt.xlim(Mmin,Mmax)
    plt.ylim(Mmin,Mmax)
    plt.legend(loc=2,fontsize=9)
    plt.tick_params(direction='in')
    plt.tight_layout(pad=0.2)
    savefigs('mdot_VIScroll_R407C')

    ## Validation injected mass flow rate
    plt.figure(figsize=(3.5,2.5))
    
    plt.plot(mdotinj_calc_ANN*1000,mdotinj_meas_SH*1000,'o',ms = 4,mfc = 'none',mec='b', label='ANN (MAPE = {:0.1f}%)'.format(MAPE_mdotinj_ANN))
    plt.plot(mdotinj_calc_ANN_2phase*1000,mdotinj_meas_2phase*1000,'d',ms = 5,mfc = 'none',mec='g', label='ANN 2$\phi$ (MAPE = {:0.1f}%)'.format(MAPE_mdotinj_ANN_2phase))
    plt.plot(mdotinj_calc_BPI*1000,mdotinj_meas_SH*1000,'s',ms = 4,mfc = 'none',mec='r', label='B-$\pi$ (MAPE = {:0.1f}%)'.format(MAPE_mdotinj_BPI))
    plt.plot(mdotinj_calc_BPI_2phase*1000,mdotinj_meas_2phase*1000,'^',ms = 5,mfc = 'none',mec='y', label='B-$\pi$ 2$\phi$ (MAPE = {:0.1f}%)'.format(MAPE_mdotinj_BPI_2phase))
    
    plt.xlabel('Inj Mass flow rate (calc) [g/s]',fontsize = 10)
    plt.ylabel('Inj Mass flow rate (meas) [g/s]',fontsize = 10)
    
    Mmin = 4
    Mmax = 30
    x=[Mmin,Mmax]
    y=[Mmin,Mmax]
    y105=[1.05*Mmin,1.05*Mmax]
    y95=[0.95*Mmin,0.95*Mmax]
    
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)
    
    plt.text(25,20,'$\pm$5%',fontsize=9)
    
    plt.legend(loc=2,fontsize=8)
    plt.xlim(Mmin,Mmax)
    plt.ylim(Mmin,Mmax)
    plt.tick_params(direction='in')
    plt.tight_layout(pad=0.2) 
    savefigs('mdotinj_VIScroll_R407C')
    
    ## Validation power output
    plt.figure(figsize=(3.5,2.5))
    
    plt.plot(Wdot_calc_ANN/1000,Wdot_meas_SH/1000,'o',ms = 4,mfc = 'none',mec='b', label='ANN (MAPE = {:0.1f}%)'.format(MAPE_Wdot_ANN))
    plt.plot(Wdot_calc_ANN_2phase/1000,Wdot_meas_2phase/1000,'d',ms = 5,mfc = 'none',mec='g', label='ANN 2$\phi$ (MAPE = {:0.1f}%)'.format(MAPE_Wdot_ANN_2phase))
    plt.plot(Wdot_calc_BPI/1000,Wdot_meas_SH/1000,'s',ms = 4,mfc = 'none',mec='r', label='B-$\pi$ (MAPE = {:0.1f}%)'.format(MAPE_Wdot_BPI))
    plt.plot(Wdot_calc_BPI_2phase/1000,Wdot_meas_2phase/1000,'^',ms = 5,mfc = 'none',mec='y', label='B-$\pi$ 2$\phi$ (MAPE = {:0.1f}%)'.format(MAPE_Wdot_BPI_2phase))

    plt.xlabel('Input Power (calc) [kW]',fontsize = 10)
    plt.ylabel('Input Power (meas) [kW]',fontsize = 10)
    
    Wmin = 0
    Wmax = 5000
    x=[Wmin/1000,Wmax/1000]
    y=[Wmin/1000,Wmax/1000]
    y105=[1.02*Wmin/1000,1.02*Wmax/1000]
    y95=[0.98*Wmin/1000,0.98*Wmax/1000]
    
    plt.text(4.7,4.3,'$\pm$2%',fontsize=9)
    
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)
    plt.xlim(3,5)
    plt.ylim(3,5)
    plt.legend(loc=4, fontsize=8)
    plt.tick_params(direction='in')
    plt.tight_layout(pad=0.2) 
    savefigs('Wdot_VIScroll_R407C')

    ## Validation isentropic efficiency
    plt.figure(figsize=(3.5,2.5))
    
    plt.plot(etais_calc_ANN,etais_meas_SH,'o',ms = 4,mfc = 'none',mec='b', label='ANN (MAPE = {:0.1f}%)'.format(MAPE_etais_ANN))
    plt.plot(etais_calc_ANN_2phase,etais_meas_2phase,'d',ms = 5,mfc = 'none',mec='g', label='ANN 2$\phi$ (MAPE = {:0.1f}%)'.format(MAPE_etais_ANN_2phase))
    plt.plot(etais_calc_BPI,etais_meas_SH,'s',ms = 4,mfc = 'none',mec='r', label='B-$\pi$ (MAPE = {:0.1f}%)'.format(MAPE_etais_BPI))
    plt.plot(etais_calc_BPI_2phase,etais_meas_2phase,'^',ms = 5,mfc = 'none',mec='y', label='B-$\pi$ 2$\phi$ (MAPE = {:0.1f}%)'.format(MAPE_etais_BPI_2phase))
    
    plt.xlabel('Isentropic Efficiency (calc) [-]',fontsize = 10)
    plt.ylabel('Isentropic Efficiency (meas) [-]',fontsize = 10)
    
    eta_min = 0
    eta_max = 0.8
    x=[eta_min,eta_max]
    y=[eta_min,eta_max]
    y105=[1.02*eta_min,1.02*eta_max]
    y95=[0.98*eta_min,0.98*eta_max]
    
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)
    
    plt.text(0.68,0.65,'$\pm$2%',fontsize=9)
    
    plt.legend(loc=2,fontsize=8)
    plt.xlim(0.6,0.7)
    plt.ylim(0.6,0.7)
    plt.tick_params(direction='in')
    plt.tight_layout(pad=0.2)
    savefigs('etas_VIScroll_R407C')

    ## Validation Heat losses
    plt.figure(figsize=(3.5,2.5))
    
    plt.plot(fq_calc_ANN,fq_meas_SH,'o',ms = 4,mfc = 'none',mec='b', label='ANN (MAPE = {:0.1f}%)'.format(MAPE_fq_ANN))
    plt.plot(fq_calc_ANN_2phase,fq_meas_2phase,'d',ms = 5,mfc = 'none',mec='g', label='ANN 2$\phi$ (MAPE = {:0.1f}%)'.format(MAPE_fq_ANN_2phase))
    plt.plot(fq_calc_BPI,fq_meas_SH,'s',ms = 4,mfc = 'none',mec='r', label='B-$\pi$ (MAPE = {:0.1f}%)'.format(MAPE_fq_BPI))
    plt.plot(fq_calc_BPI_2phase,fq_meas_2phase,'^',ms = 5,mfc = 'none',mec='y', label='B-$\pi$ 2$\phi$ (MAPE = {:0.1f}%)'.format(MAPE_fq_BPI_2phase))
        
    plt.xlabel('Heat Loss Fraction (calc) [-]',fontsize = 10)
    plt.ylabel('Heat Loss Fraction (meas) [-]',fontsize = 10)
    
    fqmin = 0
    fqmax = 0.12
    x=[fqmin,fqmax]
    y=[fqmin,fqmax]
    y105=[1.1*fqmin,1.1*fqmax]
    y95=[0.9*fqmin,0.9*fqmax]
    
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)
    
    plt.text(0.08,0.065,'$\pm$10%',fontsize=9)
    
    plt.xlim(0,0.12)
    plt.ylim(0,0.12)
    plt.legend(loc=2,fontsize=8)
    plt.tick_params(direction='in')
    plt.tight_layout(pad=0.2) 
    savefigs('fq_VIScroll_R407C')

    
if __name__=='__main__':  
    
    
    # Read excel file scroll expander
    filename = os.path.dirname(__file__)+'/OilFreeScroll_Data.xlsx'
    df_meas = pd.read_excel(open(filename,'rb'), sheet_name='Experimental_Data',skipfooter=0)
    df_calc_ANN = pd.read_excel(open(filename,'rb'), sheet_name='ANN_Validation',skipfooter=0)
    df_calc_SemiEmp = pd.read_excel(open(filename,'rb'), sheet_name='SemiEmp_Validation',skipfooter=0)
    
    # Read excel file VI scroll compressor
    filename = os.path.dirname(__file__)+'/VIScroll_Data.xlsx'
    df_meas_comp_SH = pd.read_excel(open(filename,'rb'), sheet_name='Experimental_Data_SH',skipfooter=0)
    df_meas_comp_2phase = pd.read_excel(open(filename,'rb'), sheet_name='Experimental_Data_2phase',skipfooter=0)
    df_calc_ANN_comp = pd.read_excel(open(filename,'rb'), sheet_name='ANN_Validation',skipfooter=0)
    df_calc_ANN_2phase_comp = pd.read_excel(open(filename,'rb'), sheet_name='ANN_Validation_2phase',skipfooter=0)    
    df_calc_BPI_comp = pd.read_excel(open(filename,'rb'), sheet_name='BiPI_Validation',skipfooter=0)
    df_calc_BPI_2phase_comp = pd.read_excel(open(filename,'rb'), sheet_name='BiPI_Validation_2phase',skipfooter=0)
         
    #ParityPlots_ScrollExpander(df_calc_ANN,df_calc_SemiEmp,df_meas)
    ParityPlots_VIScrollComp(df_calc_ANN_comp,df_calc_ANN_2phase_comp,df_calc_BPI_comp,df_calc_BPI_2phase_comp,df_meas_comp_SH,df_meas_comp_2phase)
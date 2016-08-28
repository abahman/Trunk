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

params = {'axes.labelsize': 10,
          'axes.linewidth':0.5,
          'font.size': 10,
          'legend.fontsize': 8,
          'legend.labelspacing':0.2,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'lines.linewidth': 0.5,
          'text.usetex': False,
          'font.family':'Times New Roman'}
pylab.rcParams.update(params)


############# Evaporator parity plot ####################

r = pd.read_excel('tuning_exp_interleaved.xlsx') #file name
df = pd.read_csv('60K-6Circuit_airMD_Ammar_Tuning_exp_interleaved_sim.csv', sep = ',', header =1)

mdot_exp=np.array(r[-1::-1]['m_r'], dtype=float) # selcect the mass flow rate backward because we need to start from test#1
mdot_model=np.array(df[3::3]['m_dot Total'], dtype=float) #only select the values for interleaved options

Q_exp = np.array(r[-1::-1]['Q'], dtype=float)
Q_model = np.array(df[3::3]['Q Total'], dtype=float)/1000 # divide by 1000 to convert to kW

f=pylab.figure(figsize=(3.5,3.5))
ax=f.add_axes((0.18,0.15,0.77,0.8))

w=0.20 #Error
ax_max = 0.2 #x and y-axes max scale tick
upp_txt = ax_max / 2.2 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = ax_max / 2.0 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'-%0.0f%%' %(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+%0.0f%%' %(w*100),ha='right',va='bottom')
##########
#comment out if second error line is not required
# w1=0.10 #Error
# upp_txt = ax_max / 1.3 #location of upper error text on plot -- adjust the number to adjust the location
# low_txt = ax_max / 1.15 #location of lower error text on plot -- adjust the number to adjust the location
# ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w1)],'k--',lw=1)
# ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w1)],'k--',lw=1)
# ax.text(low_txt-0.002,low_txt*(1-w1),'-%0.0f%%' %(w1*100),ha='left',va='top')
# ax.text(upp_txt-0.002,upp_txt*(1+w1),'+%0.0f%%' %(w1*100),ha='right',va='bottom')
#########
ax.set_xlabel('$\dot m_{exp}$ [kg/s]')
ax.set_ylabel('$\dot m_{model}$ [kg/s]')
ax.plot(mdot_exp,mdot_model,'o',ms=4,markerfacecolor='None',label='Mass flow rate',mec='b',mew=1)
leg=ax.legend(loc='upper left',numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
ax.set_xlim((0,ax_max))
ax.set_ylim((0,ax_max))
pylab.savefig('images_interleaved/Evap_parity_interleave_mass.pdf')
pylab.show()
pylab.close()


#####cooling capacity plot#####
f=pylab.figure(figsize=(3.5,3.5))
ax=f.add_axes((0.18,0.15,0.77,0.8))

w=0.20 #Error
ax_max = 30 #x and y-axes max scale tick
upp_txt = ax_max / 1.8 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = ax_max / 1.3 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'-%0.0f%%' %(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+%0.0f%%' %(w*100),ha='right',va='bottom')
ax.set_xlabel('$\dot Q_{exp}$ [kW]')
ax.set_ylabel('$\dot Q_{model}$ [kW]')
ax.plot(Q_exp,Q_model,'s',ms=4,markerfacecolor='None',label='Cooling capacity',mec='r',mew=1)
leg=ax.legend(loc='upper left',numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
ax.set_xlim((0,ax_max))
ax.set_ylim((0,ax_max))
pylab.savefig('images_interleaved/Evap_parity_interleave_Q.pdf')
pylab.show()
pylab.close()

#########################
#####Combined plot#######
#########################
m_mean=np.mean(mdot_exp)
Q_mean=np.mean(Q_exp)

f=pylab.figure(figsize=(3.5,3.5))
ax=f.add_axes((0.18,0.15,0.77,0.8))

w=0.20 #Error
ax_max = 2 #x and y-axes max scale tick
upp_txt = ax_max / 1.8 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = ax_max / 1.3 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'-%0.0f%%' %(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+%0.0f%%' %(w*100),ha='right',va='bottom')
ax.set_xlabel('Normalized experiment value')
ax.set_ylabel('Normalized model value')
ax.plot(mdot_exp/m_mean,mdot_model/m_mean,'o',ms=4,markerfacecolor='None',label='Mass flow rate',mec='b',mew=1)
ax.plot(Q_exp/Q_mean,Q_model/Q_mean,'s',ms=4,markerfacecolor='None',label='Cooling capacity',mec='r',mew=1)
leg=ax.legend(loc='upper left',numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
ax.set_xlim((0,ax_max))
ax.set_ylim((0,ax_max))
pylab.savefig('images_interleaved/Evap_parity_interleaved_combined.pdf')
pylab.show()
pylab.close()



################################################################################
############# Backward Baseline prediction Evaporator parity plot ##############
################################################################################

r = pd.read_excel('tuning_exp_baseline.xlsx') #file name
df = pd.read_csv('60K-6Circuit_airMD_Ammar_Tuning_exp_interleaved_sim.csv', sep = ',', header =1)

mdot_exp=np.array(r[-1::-1]['m_r'], dtype=float) # selcect the mass flow rate backward because we need to start from test#1
mdot_model=np.array(df[2::3]['m_dot Total'], dtype=float) #only select the values for interleaved options

Q_exp = np.array(r[-1::-1]['Q'], dtype=float)
Q_model = np.array(df[2::3]['Q Total'], dtype=float)/1000 # divide by 1000 to convert to kW

f=pylab.figure(figsize=(3.5,3.5))
ax=f.add_axes((0.18,0.15,0.77,0.8))

w=0.20 #Error
ax_max = 0.2 #x and y-axes max scale tick
upp_txt = ax_max / 2.2 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = ax_max / 2.0 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'-%0.0f%%' %(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+%0.0f%%' %(w*100),ha='right',va='bottom')
##########
#comment out if second error line is not required
# w1=0.10 #Error
# upp_txt = ax_max / 1.3 #location of upper error text on plot -- adjust the number to adjust the location
# low_txt = ax_max / 1.15 #location of lower error text on plot -- adjust the number to adjust the location
# ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w1)],'k--',lw=1)
# ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w1)],'k--',lw=1)
# ax.text(low_txt-0.002,low_txt*(1-w1),'-%0.0f%%' %(w1*100),ha='left',va='top')
# ax.text(upp_txt-0.002,upp_txt*(1+w1),'+%0.0f%%' %(w1*100),ha='right',va='bottom')
#########
ax.set_xlabel('$\dot m_{exp}$ [kg/s]')
ax.set_ylabel('$\dot m_{model}$ [kg/s]')
ax.plot(mdot_exp,mdot_model,'o',ms=4,markerfacecolor='None',label='Mass flow rate',mec='b',mew=1)
leg=ax.legend(loc='upper left',numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
ax.set_xlim((0,ax_max))
ax.set_ylim((0,ax_max))
pylab.savefig('images_interleaved/Evap_parity_interleave_mass_PredictBaseline.pdf')
pylab.show()
pylab.close()


#####cooling capacity plot#####
f=pylab.figure(figsize=(3.5,3.5))
ax=f.add_axes((0.18,0.15,0.77,0.8))

w=0.20 #Error
ax_max = 30 #x and y-axes max scale tick
upp_txt = ax_max / 1.8 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = ax_max / 1.3 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'-%0.0f%%' %(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+%0.0f%%' %(w*100),ha='right',va='bottom')
ax.set_xlabel('$\dot Q_{exp}$ [kW]')
ax.set_ylabel('$\dot Q_{model}$ [kW]')
ax.plot(Q_exp,Q_model,'s',ms=4,markerfacecolor='None',label='Cooling capacity',mec='r',mew=1)
leg=ax.legend(loc='upper left',numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
ax.set_xlim((0,ax_max))
ax.set_ylim((0,ax_max))
pylab.savefig('images_interleaved/Evap_parity_interleave_Q_PredictBaseline.pdf')
pylab.show()
pylab.close()

#########################
#####Combined plot#######
#########################
m_mean=np.mean(mdot_exp)
Q_mean=np.mean(Q_exp)

f=pylab.figure(figsize=(3.5,3.5))
ax=f.add_axes((0.18,0.15,0.77,0.8))

w=0.20 #Error
ax_max = 2 #x and y-axes max scale tick
upp_txt = ax_max / 1.8 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = ax_max / 1.3 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'-%0.0f%%' %(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+%0.0f%%' %(w*100),ha='right',va='bottom')
ax.set_xlabel('Normalized experiment value')
ax.set_ylabel('Normalized model value')
ax.plot(mdot_exp/m_mean,mdot_model/m_mean,'o',ms=4,markerfacecolor='None',label='Mass flow rate',mec='b',mew=1)
ax.plot(Q_exp/Q_mean,Q_model/Q_mean,'s',ms=4,markerfacecolor='None',label='Cooling capacity',mec='r',mew=1)
leg=ax.legend(loc='upper left',numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
ax.set_xlim((0,ax_max))
ax.set_ylim((0,ax_max))
pylab.savefig('images_interleaved/Evap_parity_interleaved_combined_PredictBaseline.pdf')
pylab.show()
pylab.close()
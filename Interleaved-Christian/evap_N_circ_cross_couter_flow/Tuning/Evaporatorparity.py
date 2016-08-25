import matplotlib,os
#matplotlib.use('GTKAgg')
import sys, os
#from FileIO import prep_csv2rec as prep
from matplotlib.mlab import csv2rec

import pylab
import numpy as np
import shutil
from scipy import polyval, polyfit

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

r=csv2rec('ExperimentalData.csv',delimiter=',') #file name

mdot_exp=np.array(r[1:]['mdot_oil_hot'], dtype=float)
mdot_model=np.array(r[1:]['mdot_oil_motor'], dtype=float)
Q_exp = 0
Q_model = 0

f=pylab.figure(figsize=(3.5,3.5))
ax=f.add_axes((0.18,0.15,0.77,0.8))

w=0.30 #Error
ax_max = 0.2 #x and y-axes max scale tick
upp_txt = ax_max / 1.8 #location of upper error text on plot -- adjust the number to adjust the location
low_txt = ax_max / 1.3 #location of lower error text on plot -- adjust the number to adjust the location
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max],'k-',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1-w)],'k-.',lw=1)
ax.plot(np.r_[0,ax_max],np.r_[0,ax_max*(1+w)],'k-.',lw=1)
ax.text(low_txt-0.002,low_txt*(1-w),'-%0.0f%%' %(w*100),ha='left',va='top')
ax.text(upp_txt-0.002,upp_txt*(1+w),'+%0.0f%%' %(w*100),ha='right',va='bottom')

ax.set_xlabel('$\dot m_{exp}$ [kg/s]')
ax.set_ylabel('$\dot m_{model}$ [kg/s]')

ax.plot(mdot_exp,mdot_model,'o',ms=4,markerfacecolor='None',label='Mass flow rate',mec='b',mew=1)
#ax.plot(mdot_comp,mdot_HX,'s',ms=4,markerfacecolor='None',label='Hot HX Energy Balance',mec='r',mew=1)
leg=ax.legend(loc='upper left',numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)

ax.set_xlim((0,ax_max))
ax.set_ylim((0,ax_max))

pylab.savefig('images/Evap_parity.pdf')
pylab.show()


################## compressor parity #######################

#r=csv2rec('Results.csv', delimiter=',')
#
#mdotmean=np.mean(r['mdot_experkgs'])
#Wdotmean=np.mean(r['p_model_kw']) #notice that the bracket [] , dashes / and \ are removed. The spaces are converted to underscore. Upper case letters converted to lower case ketters.
#
#f=pylab.figure(figsize=(3.5,3.5))
#ax=f.add_axes((0.15,0.15,0.8,0.8))
#
#w=0.03
#ax.plot(np.r_[0,2],np.r_[0,2],'k-.',lw=1)
#ax.plot(np.r_[0,2],np.r_[0,2*(1+w)],'k-',lw=1)
#ax.plot(np.r_[0,2],np.r_[0,2*(1-w)],'k-',lw=1)
#ax.text(1.7,1.7*(1-w),'-%0.0f%%' %(w*100),ha='left',va='top')
#ax.text(1.7-0.02,1.7*(1+w),'+%0.0f%%' %(w*100),ha='right',va='bottom')
#ax.set_xlabel('Normalized experiment value')
#ax.set_ylabel('Normalized model value')
#
#ax.plot(r['mdot_model_kgs']/mdotmean,r['mdot_experkgs']/mdotmean,'o',ms=4,markerfacecolor='None',label='Mixture Mass Flow Rate',mec='b',mew=1)
#ax.plot(r['p_model_kw']/Wdotmean,r['p_exper_kw']/Wdotmean,'s',ms=4,markerfacecolor='None',label='Shaft Power',mec='r',mew=1)
#leg=ax.legend(loc='upper left',numpoints=1)
#frame  = leg.get_frame()
#frame.set_linewidth(0.5)
#
#ax.set_xlim((0,2))
#ax.set_ylim((0,2))
#
#pylab.savefig('images/CompressorParityPlot.pdf')
##pylab.savefig('images/CompressorParityPlot.eps')
##pylab.savefig('images/CompressorParityPlot.png',dpi=600)
#pylab.show()
#pylab.close()



################## Expander parity #######################

#r=loadmat('results100301.mat',struct_as_record =True)
#print r.keys()
#
#mdotmean=np.mean(r['M_dot_meas'][0,0:26])
#Wdotmean=np.mean(r['W_dot_sh_meas'][0,0:26])
#
#f=pylab.figure(figsize=(3.5,3.5))
#ax=f.add_axes((0.15,0.15,0.8,0.8))
#
#w=0.1
#ax.plot(np.r_[0,2],np.r_[0,2],'k-.',lw=1)
#ax.plot(np.r_[0,2],np.r_[0,2*(1+w)],'k-',lw=1)
#ax.plot(np.r_[0,2],np.r_[0,2*(1-w)],'k-',lw=1)
#ax.text(1.7,1.7*(1-w),'-%0.0f%%' %(w*100),ha='left',va='top')
#ax.text(1.7-0.02,1.7*(1+w),'+%0.0f%%' %(w*100),ha='right',va='bottom')
#ax.set_xlabel('Normalized experiment value')
#ax.set_ylabel('Normalized model value')
#
#xnorm=np.array(r['M_dot_calc'][0,0:26]/mdotmean)
#ynorm=np.array(r['M_dot_meas'][0,0:26]/mdotmean)
#s1=ax.plot(xnorm,ynorm,'o',ms=4,markerfacecolor='None',label='Mixture Mass Flow Rate',mec='b',mew=1)
#
#xnorm=r['W_dot_sh_calc'][0,0:26]/Wdotmean
#ynorm=r['W_dot_sh_meas'][0,0:26]/Wdotmean
#s2=ax.plot(xnorm,ynorm,'s',ms=4,markerfacecolor='None',label='Shaft Power',mec='r',mew=1)
#
#leg=ax.legend(loc='upper left',numpoints=1)
#frame  = leg.get_frame()
#frame.set_linewidth(0.5)
#
#ax.set_xlim((0,2))
#ax.set_ylim((0,2))
#
#pylab.savefig('images/ExpanderParityPlot.pdf')
##pylab.savefig('images/ExpanderParityPlot.eps')
##pylab.savefig('images/ExpanderParityPlot.png',dpi=600)
#pylab.show()
#pylab.close()
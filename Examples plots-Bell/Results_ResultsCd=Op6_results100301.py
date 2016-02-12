import matplotlib,os
#matplotlib.use('GTKAgg')
import sys,os
#from FileIO.prep_csv2rec import prep_csv2rec
from matplotlib.mlab import csv2rec

import pylab
import numpy as np
import shutil
from scipy import polyval, polyfit
from scipy.io import loadmat

params = {'axes.labelsize': 10,
          'axes.linewidth': 0.5,
          'font.size': 10,
          'legend.fontsize': 8,
          'legend.labelspacing':0.2,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'lines.linewidth': 0.5,
          'text.usetex': False,
          'font.family':'Times New Roman'}
pylab.rcParams.update(params)

r=csv2rec('Results.csv', delimiter=',')

mdotmean=np.mean(r['mdot_experkgs'])
Wdotmean=np.mean(r['p_model_kw']) #notice that the bracket [] , dashes / and \ are removed. The spaces are converted to underscore. Upper case letters converted to lower case ketters.

f=pylab.figure(figsize=(3.5,3.5))
ax=f.add_axes((0.15,0.15,0.8,0.8))
 
w=0.03
ax.plot(np.r_[0,2],np.r_[0,2],'k-.',lw=1)
ax.plot(np.r_[0,2],np.r_[0,2*(1+w)],'k-',lw=1)
ax.plot(np.r_[0,2],np.r_[0,2*(1-w)],'k-',lw=1)
ax.text(1.7,1.7*(1-w),'-%0.0f%%' %(w*100),ha='left',va='top')
ax.text(1.7-0.02,1.7*(1+w),'+%0.0f%%' %(w*100),ha='right',va='bottom')
ax.set_xlabel('Normalized experiment value')
ax.set_ylabel('Normalized model value')
 
ax.plot(r['mdot_model_kgs']/mdotmean,r['mdot_experkgs']/mdotmean,'o',ms=4,markerfacecolor='None',label='Mixture Mass Flow Rate',mec='b',mew=1)
ax.plot(r['p_model_kw']/Wdotmean,r['p_exper_kw']/Wdotmean,'s',ms=4,markerfacecolor='None',label='Shaft Power',mec='r',mew=1)
leg=ax.legend(loc='upper left',numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
 
ax.set_xlim((0,2))
ax.set_ylim((0,2))
 
pylab.savefig('images/CompressorParityPlot.pdf')
#pylab.savefig('images/CompressorParityPlot.eps')
#pylab.savefig('images/CompressorParityPlot.png',dpi=600)
pylab.show()
pylab.close()
 
f=pylab.figure(figsize=(3.5,3.5))
ax=f.add_axes((0.15,0.15,0.8,0.8))
ax.plot(r['xl_'],r['mdot_model_kgs']/r['mdot_experkgs'],'o',mfc='none',label=r'$\dot m_m$')
ax.plot(r['xl_'],r['p_model_kw']/r['p_exper_kw'],'*',mfc='none',label=r'$\dot W_{shaft}$')
ax.set_xlabel('$x_l$ [-]',size=10)
ax.set_ylabel('Model/Experimental [-]',size=10)
ax.legend(loc='best',numpoints=1)
ax.axhline(1.0,ls='--')
ax.set_ylim((0.95,1.05))
pylab.savefig('images/CompressorParityPlot2.pdf')
#pylab.savefig('images/CompressorParityPlot.eps')
#pylab.savefig('images/CompressorParityPlot.png',dpi=600)
pylab.show()
pylab.close()
 
## shutil.copy('CompressorParityPlot.eps',os.path.join('..','..','TeX','CompParity.eps'))
 
############# ML fit ############
 
p_gas=r['p_gas_kw']

p_MLExper=r['p_exper_kw']-r['p_gas_kw']
m=r['m_kw'][0]
b=r['b_'][0]
 
f=pylab.figure(figsize=(3.5,3.5))
ax=f.add_axes((0.18,0.15,0.75,0.8))
ax.plot(p_gas,p_MLExper,'o',ms=6,markerfacecolor='None',mec='b',mew=1)
p=np.linspace(np.min(p_gas),np.max(p_gas),2)
ax.plot(p,m*p+b,'k-',ms=6,markerfacecolor='None',label='Shaft Power',mec='r',mew=1,lw=1)
ax.set_ylabel('$\dot W_{ML}$ [kW]',size=10)
ax.set_xlabel('$\dot W_{gas}$ [kW]',size=10)
 
pylab.savefig('images/MLfit.pdf')
#pylab.savefig('images/MLfit.eps')
#pylab.savefig('images/MLfit.png',dpi=600)
pylab.show()
pylab.close()
 
## shutil.copy('MLfit.eps',os.path.join('..','..','TeX','MLfit.eps'))


############# xL dependence ###############
q=csv2rec('ResultsCd=0p6.csv',delimiter=',')

xL=q['xl_']
XL=np.linspace(np.min(xL),np.max(xL),2)
p_MLExperCd=q['p_exper_kw']-q['p_gas_kw']

f=pylab.figure(figsize=(3.5,3.5))
ax=f.add_axes((0.18,0.15,0.75,0.8))

ax.plot(xL,p_MLExper,'ro',ms=6,markerfacecolor='None',mec='r',mew=1,label='$C_{d,disc}=0.4$')
(ar,br)=polyfit(xL,p_MLExper,1)
ax.plot(XL,polyval([ar,br],XL),'r-')

ax.plot(xL,p_MLExperCd,'bx',ms=6,markerfacecolor='None',mec='b',mew=1,label='$C_{d,disc}=0.6$')
(ar,br)=polyfit(xL,p_MLExperCd,1)
ax.plot(XL,polyval([ar,br],XL),'b-')

ax.set_xlabel('$x_l$ [kW]',size=10)
ax.set_ylabel('$\dot W_{ML}$ [kW]',size=10)
ax.legend(numpoints=1,loc='lower center')

pylab.savefig('images/MLxLdep.pdf')
#pylab.savefig('images/MLxLdep.eps')
#pylab.savefig('images/MLxLdep.png',dpi=600)
pylab.show()
pylab.close()

## shutil.copy('MLxLdep.eps',os.path.join('..','..','TeX','MLxLdep.eps'))

################## Expander parity #######################

r=loadmat('results100301.mat',struct_as_record =True)
print r.keys()

mdotmean=np.mean(r['M_dot_meas'][0,0:26])
Wdotmean=np.mean(r['W_dot_sh_meas'][0,0:26])

f=pylab.figure(figsize=(3.5,3.5))
ax=f.add_axes((0.15,0.15,0.8,0.8))

w=0.1
ax.plot(np.r_[0,2],np.r_[0,2],'k-.',lw=1)
ax.plot(np.r_[0,2],np.r_[0,2*(1+w)],'k-',lw=1)
ax.plot(np.r_[0,2],np.r_[0,2*(1-w)],'k-',lw=1)
ax.text(1.7,1.7*(1-w),'-%0.0f%%' %(w*100),ha='left',va='top')
ax.text(1.7-0.02,1.7*(1+w),'+%0.0f%%' %(w*100),ha='right',va='bottom')
ax.set_xlabel('Normalized experiment value')
ax.set_ylabel('Normalized model value')

xnorm=np.array(r['M_dot_calc'][0,0:26]/mdotmean)
ynorm=np.array(r['M_dot_meas'][0,0:26]/mdotmean)
s1=ax.plot(xnorm,ynorm,'o',ms=4,markerfacecolor='None',label='Mixture Mass Flow Rate',mec='b',mew=1)

xnorm=r['W_dot_sh_calc'][0,0:26]/Wdotmean
ynorm=r['W_dot_sh_meas'][0,0:26]/Wdotmean
s2=ax.plot(xnorm,ynorm,'s',ms=4,markerfacecolor='None',label='Shaft Power',mec='r',mew=1)

leg=ax.legend(loc='upper left',numpoints=1)
frame  = leg.get_frame()
frame.set_linewidth(0.5)

ax.set_xlim((0,2))
ax.set_ylim((0,2))

pylab.savefig('images/ExpanderParityPlot.pdf')
#pylab.savefig('images/ExpanderParityPlot.eps')
#pylab.savefig('images/ExpanderParityPlot.png',dpi=600)
pylab.show()
pylab.close()

f=pylab.figure(figsize=(3.5,3.5))
ax=f.add_axes((0.15,0.15,0.8,0.8))
ax.plot(r['M_dot_meas'][0,0:26],r['M_dot_calc'][0,0:26]/r['M_dot_meas'][0,0:26],'o',mfc='none',label=r'$\dot m_m$')
ax.plot(r['M_dot_meas'][0,0:26],r['W_dot_sh_calc'][0,0:26]/r['W_dot_sh_meas'][0,0:26],'*',mfc='none',label=r'$\dot W_{shaft}$')
ax.set_xlabel('$\dot m_{m}$ experimental [kg/s]',size=10)
ax.set_ylabel('Model/Experimental [-]',size=10)
ax.set_ylim((0.90,1.1))
ax.axhline(1.0,ls='--')
ax.legend(loc='best',numpoints=1)
pylab.savefig('images/ExpanderParityPlot2.pdf')
#pylab.savefig('images/ExpanderParityPlot2.eps')
#pylab.savefig('images/ExpanderParityPlot2.png',dpi=600)
pylab.show()
pylab.close()
## shutil.copy('CompressorParityPlot.eps',os.path.join('..','..','TeX','CompParity.eps'))

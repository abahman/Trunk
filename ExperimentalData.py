import matplotlib,os
matplotlib.use('GTKAgg')
import sys
from FileIO import prep_csv2rec as prep
from matplotlib.mlab import csv2rec

import pylab
import numpy as np
import shutil
from scipy import polyval, polyfit

params = {'axes.labelsize': 10,
          'axes.linewidth':0.5,
          'text.fontsize': 10,
          'legend.fontsize': 8,
          'legend.labelspacing':0.2,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'lines.linewidth': 0.5,
          'text.usetex': False,
          'font.family':'Times New Roman'}
pylab.rcParams.update(params)

r=prep('ExperimentalData.csv',delRowList=[1])

############# HOT SIDE ####################
mdot_comp=r['mdot_oil_hot']
mdot_motor=r['mdot_oil_motor']
mdot_HX=r['mdot_oil_hothx']

f=pylab.figure(figsize=(3.5,3.5))
ax=f.add_axes((0.18,0.15,0.77,0.8))

w=0.30
ax.plot(np.r_[0,0.2],np.r_[0,0.2],'k-',lw=1)
ax.plot(np.r_[0,0.2],np.r_[0,0.2*(1-0.3)],'k-.',lw=1)
ax.plot(np.r_[0,0.2],np.r_[0,0.2*(1+0.3)],'k-.',lw=1)
ax.text(0.17-0.002,0.17*(1-w),'-%0.0f%%' %(w*100),ha='left',va='top')
ax.text(0.1-0.002,0.1*(1+w),'+%0.0f%%' %(w*100),ha='right',va='bottom')

ax.set_xlabel('$\dot m_{oil}$ from compressor energy balance [kg/s]')
ax.set_ylabel('$\dot m_{oil}$ ')

ax.plot(mdot_comp,mdot_motor,'o',ms=4,markerfacecolor='None',label='Hyd. Exp. Displacement Rate',mec='b',mew=1)
ax.plot(mdot_comp,mdot_HX,'s',ms=4,markerfacecolor='None',label='Hot HX Energy Balance',mec='r',mew=1)
leg=ax.legend(loc='upper left',numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)

ax.set_xlim((0,0.2))
ax.set_ylim((0,0.2))

pylab.savefig('HotOilMdot.pdf')
pylab.savefig('HotOilMdot.eps')
pylab.savefig('HotOilMdot.png',dpi=600)
pylab.show()

shutil.copy('HotOilMdot.eps',os.path.join('..','TeX','HotOilMdot.eps'))

############# COLD SIDE ####################
mdot_exp=r['mdot_oil_cold']
mdot_pump=r['mdot_oil_pump']
mdot_HX=r['mdot_oil_coldhx']

f=pylab.figure(figsize=(3.5,3.5))
ax=f.add_axes((0.18,0.15,0.77,0.8))

w=0.40
ax.plot(np.r_[0,0.1],np.r_[0,0.1],'k-',lw=1)
ax.plot(np.r_[0,0.1],np.r_[0,0.1*(1.0-w)],'k-.',lw=1)
ax.plot(np.r_[0,0.1],np.r_[0,0.1*(1.0+w)],'k-.',lw=1)

ax.text(0.07-0.002,0.07*(1-w),'-%0.0f%%' %(w*100),ha='left',va='top')
ax.text(0.05-0.002,0.05*(1+w),'+%0.0f%%' %(w*100),ha='right',va='bottom')

ax.set_xlabel('$\dot m_{oil}$ from expander energy balance [kg/s]')
ax.set_ylabel('$\dot m_{oil}$ ')

ax.plot(mdot_exp,mdot_pump,'o',ms=4,markerfacecolor='None',label='Pump Displacement Rate',mec='b',mew=1)
ax.plot(mdot_exp,mdot_HX,'s',ms=4,markerfacecolor='None',label='Cold HX Energy Balance',mec='r',mew=1)
leg=ax.legend(loc='upper left',numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)

ax.set_xlim((0,0.1))
ax.set_ylim((0,0.1))

pylab.savefig('ColdOilMdot.pdf')
pylab.savefig('ColdOilMdot.eps')
pylab.savefig('ColdOilMdot.png',dpi=600)
pylab.show()

shutil.copy('ColdOilMdot.eps',os.path.join('..','TeX','ColdOilMdot.eps'))
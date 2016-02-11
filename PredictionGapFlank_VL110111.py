import matplotlib
#matplotlib.use('GTKAgg')
import sys,os
#from FileIO import prep_csv2rec
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

r=csv2rec('PredictionGapFlank_VL110111.csv', delimiter=',')

gap_meas=r['gap_meas']*10**6
gap_calc=r['gap_calc']*10**6

print gap_meas,'\n\n', gap_calc

f=pylab.figure(figsize=(3.5,3.5))
ax=f.add_axes((0.15,0.15,0.8,0.8))

w=0.25
m=100
ax.plot(np.r_[0,m],np.r_[0,m],'k-.',lw=1)
ax.plot(np.r_[0,m],np.r_[0,m*(1+w)],'k-',lw=1)
ax.plot(np.r_[0,m],np.r_[0,m*(1-w)],'k-',lw=1)
ax.text(60,60*(1-w),'-%0.0f%%' %(w*100),ha='left',va='top')
ax.text(60-1,60*(1+w),'+%0.0f%%' %(w*100),ha='right',va='bottom')

ax.set_ylabel('$\delta_{f,correlation}$ [$\mu$ m]')
ax.set_xlabel('$\delta_{f,identified}$ [$\mu$ m]')

ax.plot(gap_meas,gap_calc,'ko',ms=4)

ax.set_xlim((0,m))
ax.set_ylim((0,m))

pylab.savefig('DeltaPlot.pdf')
#pylab.savefig('DeltaPlot.eps')
#pylab.savefig('DeltaPlot.png',dpi=600)
pylab.show()
pylab.close()

## shutil.copy('CompressorParityPlot.eps',os.path.join('..','..','TeX','CompParity.eps'))

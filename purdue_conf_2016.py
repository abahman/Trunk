from __future__ import division
'''
Created on Mar 9, 2016

@author: AmmarBahman
'''
'''This code is to plot all figure for Purdue conference 2016'''

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.Plots import PropertyPlot
from CoolProp.CoolProp import PropsSI
import CoolProp
from matplotlib.ticker import ScalarFormatter

import matplotlib as mpl
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
"xtick.labelsize": 8,
"ytick.labelsize": 8,
"figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
"pgf.preamble": [
r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)


P_18k = np.array([655.8,3108.0,3108.0,3108.0,3095.0,3095.0,3080.0,975.0,655.8,655.8,655.8]) #in kPa 
h_18k = np.array([421.7,486.2,485.1,423.7,305.2,295.6,295.6,295.6,414.3,420.4,421.7]) #in kJ/kg    

P_60k = np.array([625.3,3213.0,3213.0,3213.0,3145.0,3145.0,3130.0,1072.0,625.3,625.3,625.3]) #in kPa 
h_60k = np.array([430,486.2,484,423,306.7,296.8,296.8,296.8,413.6,425.2,430]) #in kJ/kg  


P_36k = np.array([1069.0,4363.0,4363.0,4363.0,4316.0,4316.0,4301.0,1308.0,1069.0,1069.0,1069.0]) #in kPa 
h_36k = np.array([435.1,492.6,471.5,403.6,323.9,302.3,302.4,302.4,423.9,434.5,435.1]) #in kJ/kg



# plot = PropertyPlot('HEOS::R245fa', 'TS', unit_system='EUR', tp_limits='ORC')
# plot.calc_isolines(CoolProp.iQ, num=11)
# plot.calc_isolines(CoolProp.iP, iso_range=[1,50], num=10, rounding=True)
# plot.draw()
# plot.isolines.clear()
# plot.props[CoolProp.iP]['color'] = 'green'
# plot.props[CoolProp.iP]['lw'] = '0.5'
# plot.calc_isolines(CoolProp.iP, iso_range=[1,50], num=10, rounding=False)
# plot.show()


ph_plot_R410A = PropertyPlot('R410A', 'PH',unit_system='KSI',tp_limits='ACHP')

ph_plot_R410A.calc_isolines(CoolProp.iQ, num=11)

plt.plot(h_36k,P_36k,'g:', label='36K ECU')
plt.ylim(500,5000)
plt.xlim(250,500)
plt.yticks([500, 1000, 5000],
               [r'500', r'$1000$', r'$5000$'])
ph_plot_R410A.show()

#Plot P-h diagram
ph_plot_R407C = PropertyPlot('R407C', 'PH',unit_system='KSI',tp_limits='ACHP')
ph_plot_R407C.props[CoolProp.iQ]['color'] = 'black'

ph_plot_R407C.calc_isolines(CoolProp.iQ, num=11)
ph_plot_R407C.calc_isolines(CoolProp.iT, num=25)
ph_plot_R407C.draw()
ph_plot_R407C.isolines.clear()

ph_plot_R410A.props[CoolProp.iQ]['color'] = 'green'
ph_plot_R410A.calc_isolines(CoolProp.iQ, num=11)
ph_plot_R410A.draw()

ph_plot_R407C.xlabel(r'$h$ $[\mathrm{kJ/kg}]$')
ph_plot_R407C.ylabel(r'$P$ $[\mathrm{kPa}]$')
#ph_plot_R407C.grid()
plt.plot(h_60k,P_60k,'b-', label='60K ECU')
#plt.errorbar(h_60k,P_60k, yerr=0.08*P_60k)
plt.plot(h_36k,P_36k,'g:', label='36K ECU')
plt.plot(h_18k,P_18k,'r--', label='18K ECU')

plt.ylim(500,5000)
plt.xlim(250,500)
plt.yticks([500, 1000, 5000],
               [r'500', r'$1000$', r'$5000$'])
plt.annotate('R407C', xy=(300, 4000), xytext=(300, 4000))
plt.legend(loc='best',fancybox=False,frameon=False)


#ph_plot_R407C.savefig('PH.pdf')
ph_plot_R407C.show()
    
    
     
# N = 5
# menMeans = (20, 35, 30, 35, 27)
# womenMeans = (25, 32, 34, 20, 25)
# menStd = (2, 3, 4, 1, 2)
# womenStd = (3, 5, 2, 3, 3)
# ind = np.arange(N)    # the x locations for the groups
# width = 0.35       # the width of the bars: can also be len(x) sequence
# 
# p1 = plt.bar(ind, menMeans, width, color='r', yerr=menStd)
# p2 = plt.bar(ind, womenMeans, width, color='y',
#              bottom=menMeans, yerr=womenStd)
# 
# plt.ylabel('Scores')
# plt.title('Scores by group and gender')
# plt.xticks(ind + width/2., ('G1', 'G2', 'G3', 'G4', 'G5'))
# plt.yticks(np.arange(0, 81, 10))
# plt.legend((p1[0], p2[0]), ('Men', 'Women'))
# 
# plt.show()

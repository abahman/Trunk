'''
Created on Aug 28, 2017

@author: ammarbahman


'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import csv2rec
import matplotlib as mpl
mpl.style.use('classic')
mpl.style.use('Elsevier.mplstyle')
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['figure.figsize'] = [6,4]
#===============================================================================
# Latex render
#===============================================================================
#mpl.use('pgf')

# def figsize(scale):
#     fig_width_pt = 469.755                          # Get this from LaTeX using \the\textwidth
#     inches_per_pt = 1.0/72.27                       # Convert pt to inch
#     golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
#     fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
#     fig_height = fig_width*golden_mean              # height in inches
#     fig_size = [fig_width,fig_height]
#     return fig_size
# 
# pgf_with_latex = {                      # setup matplotlib to use latex for output
# "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
# "text.usetex": True,                # use LaTeX to write all text
# "font.family": "serif",
# "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
# "font.sans-serif": [],
# "font.monospace": [],
# "axes.labelsize": 10,               # LaTeX default is 10pt font.
# "font.size": 10,
# "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
# "legend.labelspacing":0.2,
# "xtick.labelsize": 8,
# "ytick.labelsize": 8,
# "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
# "pgf.preamble": [
# r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
# r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
#         ]
#     }
# mpl.rcParams.update(pgf_with_latex)
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

#Experimental Results
TestNo = np.arange(1,7,1)
cooling_capacity_exp = np.array([15.1,14.96,15.4,13.2,13.44,15.17]) #[kW]
COPS_exp = np.array([2.743,2.726,2.776,2.176,2.274,2.745]) #[-]

pressure_ratio = np.array([2.103205629,2.11023622,2.05509434,2.213523132,2.17211329,2.088666153]) #[-]
isentropic_expansion = np.array([343.5,346.2,329.2,469.6,424.6,342]) #[W]
nozzle = np.array([130.3,130.1,131.1,183.1,164.7,132.3]) #[W]
fluid = np.array([71.07,70.77,70.55,94.16,88.09,71.67]) #[W]
mechanical = np.array([66.59,66.36,65.91,86.82,81.48,67.05]) #[W]
electric = np.array([58.6,58.4,58,76.4,71.7,59]) #[W]

#===============================================================================
# #Bar plots
#===============================================================================
#COPS
fig=plt.figure(figsize=(6,4.5))
plt.bar(np.arange(1,7,1)-0.1,COPS_exp,width=0.2,color='r',linewidth=0.9,align='center',alpha=0.9,label=r'COP [$-$]',hatch=5*'\\')
plt.errorbar(np.arange(1,7,1)-0.1,COPS_exp,yerr=[0.044,0.044,0.043,0.034,0.036,0.043],capsize=2,elinewidth=0.7,fmt='',linestyle="None",color='k')
plt.bar(np.arange(1,7,1)+0.1,cooling_capacity_exp,width=0.2,color='b',linewidth=0.9,align='center',alpha=0.9,label=r'Cooling Capacity [kW]',hatch=2*'//')
plt.errorbar(np.arange(1,7,1)+0.1,cooling_capacity_exp,yerr=[0.2391,0.2384,0.2369,0.2071,0.2104,0.2379],capsize=2,elinewidth=0.7,fmt='',linestyle="None",color='k')
# plt.text(1,3.65,r'Improvment at Test 4/A = {:0.01f}\%'.format((COPS2[3]-COPS_exp[3])/COPS_exp[3] *100),ha='left',va='center',fontsize = 10)
# plt.text(1,4,r'Improvment at Test 1 = {:0.01f}\%'.format((COPS2[0]-COPS_exp[0])/COPS_exp[0] *100),ha='left',va='center',fontsize = 10)
plt.ylim(0,18)
plt.xlim(0,7)
# plt.yticks([0, 3, 6, 9, 12, 15, 18],
#             [r'0', r'3', r'6', r'9',r'12', r'15', r'18'])
plt.xticks([0, 1, 2, 3, 4, 5, 6],
            [r'', r'25/35', r'25/40', r'25/42',r'26.7/35', r'23.5/35', r'24/35'])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.xlabel(r'Test condition (indoor/outdoor) [$\degree$C]')
plt.ylabel(r'Performance')
leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout() 
plt.savefig('performance_bar.pdf')
plt.show()


#===============================================================================
#Plot_power_availability
#===============================================================================
fig=plt.figure(figsize=(6,4.5))
plt.plot(pressure_ratio, isentropic_expansion,'d',ms = 5,mfc = 'b',mec='b', label='Isentropic Expansion')
plt.plot(pressure_ratio, nozzle,'^',ms = 5,mfc = 'r',mec='r', label='Nozzle')
plt.plot(pressure_ratio, fluid,'o',ms = 5,mfc = 'k',mec='k', label='Fluid')
plt.plot(pressure_ratio, mechanical,'x',ms = 5,mfc = 'y',mec='y', label='Mechanical')
plt.plot(pressure_ratio, electric,'+',ms = 5,mfc = 'g',mec='g', label='Electric')
plt.text(2.235,140,r'Fluid Losses',ha='left',va='center',fontsize = 10)
plt.annotate(s='', xy=(2.23, 100), xytext=(2.23, 180), arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0))

plt.text(2.235,350,r'Nozzle Losses',ha='left',va='center',fontsize = 10)
plt.annotate(s='', xy=(2.23, 200), xytext=(2.23, 450), arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0))

plt.ylim(0,500)
plt.xlim(2,2.3)
# plt.xticks([0, 1, 2, 3, 4, 5, 6],
#             [r'', r'25/35', r'25/40', r'25/42',r'26.7/35', r'23.5/35', r'24/35'])
plt.xlabel(r'Pressure Ratio [$-$]')
plt.ylabel(r'Available Power [W]')
leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),fancybox=False,numpoints=1,ncol=3)
frame = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()
plt.savefig('power_available.pdf')
plt.show()



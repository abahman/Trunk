'''
Created on Sep 13, 2016

@author: ammarbahman

Note: this plots for calorimter paper

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.mlab as mlab
import matplotlib.cm as cm
from pylab import contourf, clf

#===============================================================================
# Latex render
#===============================================================================
import matplotlib as mpl
#mpl.use('pgf')

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
#===============================================================================
# END of Latex render
#===============================================================================
 


x = np.array([6.219,12.438,18.657])
y = np.array([20.625,16.875,13.125,9.375,5.625,1.875])

X, Y = np.meshgrid(x, y)
#===============================================================================
# Temperature profile for All tests
#===============================================================================
Test =['1','2','3','4','5','6','B','C']

T1 = np.matrix([[20.53,21.61,20.09],[20.27,20.76,20.41],[19.58,19.67,19.51],
                [19.69,19.74,19.78],[18.81,18.85,18.96],[17.68,17.93,18.22]])
T2 = np.matrix([[16.29,16.35,16.21],[17.18,17.33,17.16],[15.27,15.69,15.3],
                [15.21,15.58,15.4],[13.71,13.53,13.99],[12.2,12.36,12.92]])
T3 = np.matrix([[15.27,15.94,15.2],[17.15,17.54,16.9],[14.28,14.78,14.31],
                [14.12,14.66,14.45],[12.69,12.68,12.9],[11.91,11.03,11.6]])
T4 = np.matrix([[15.68,15.79,14.86],[15.64,16.08,15.95],[14.6,14.83,14.57],
                [14.7,14.86,14.65],[13.63,13.67,13.75],[11.92,12.3,12.6]])
T5 = np.matrix([[12.45,12.45,12.91],[13.69,15.29,14.65],[10.99,10.97,10.59],
               [10.76,10.72,10.58],[7.849,8.311,8.868],[4.798,5.016,6.926]])
T6 = np.matrix([[12.14,11.36,11.1],[13.17,13.61,14.02],[10.4,11.43,11.51],
               [9.983,10.64,11.01],[7.872,7.648,8.034],[5.318,5.505,5.93]])
TB = np.matrix([[14.96,15.57,14.41],[14.99,15.29,15.11],[13.91,14,13.74],
               [13.9,14.11,13.86],[12.87,12.86,13.03],[11.01,11.35,11.7]])
TC = np.matrix([[13.07,12.27,12.12],[13.94,14.49,15.14],[11.04,12.17,12.34],
               [10.83,11.54,11.93],[9.108,8.867,9.341],[6.68,6.905,7.573]])

T_data = [T1,T2,T3,T4,T5,T6,TB,TC]


x = np.random.random(50)
y = np.random.random(50)
c = np.random.random(50)  # color of points (color bar points)
s = 500 * np.random.random(50)  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)

# Add a colorbar
cbar = plt.colorbar(im, ax=ax)

# set the color limits - not necessary here, but good to know how.
im.set_clim(0.0, 1.0)
cbar.ax.set_ylabel(r'Discharge Pressure [kPa]')

#plt.ylim(0,22.5)
#plt.xlim(0,24.875)
#plt.xticks([0, 5, 10, 15, 20, 24.875],
#          [r'$0$', r'$5$', r'$10$', r'$15$', r'$20$', r'$24.875$'])
#plt.yticks([0, 4, 8, 12, 16, 20, 22.5],
#          [r'$0$', r'$4$', r'$8$', r'$12$',r'$16$', r'$20$', r'$22.5$'])
plt.xlabel('Suction pressure [kPa]')
plt.ylabel('Pressure ratio [-]')
              
plt.savefig('pressure_ratio.pdf')
plt.show()
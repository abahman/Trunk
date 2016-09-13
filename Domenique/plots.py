'''
Created on Sep 13, 2016

@author: ammarbahman

Note: this plots for calorimter paper

'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd

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

df = pd.read_excel("/Users/ammarbahman/desktop/Eclipse Mars/Test/Domenique/results.xlsx")

x = np.array(df[1:]['PE1_p_comp_suc'], dtype=float) * 6.89476 #convert from psia to kPa
y = np.array(df[1:]['PR'], dtype=float)
c = np.array(df[1:]['PE2_p_comp_dis'], dtype=float) * 6.89476  # color of points (color bar points)
s = 60  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)

# Add a colorbar
cbar = plt.colorbar(im, ax=ax)

# set the color limits
im.set_clim(1000, 2600)
cbar.ax.set_ylabel('Discharge Pressure [kPa]')
ax.text(np.mean(x)/2,np.mean(y)/2,'Marker size ()')#,ha='right',va='top')
#plt.ylim(0,22.5)
#plt.xlim(0,24.875)
plt.xlabel('Suction pressure [kPa]')
plt.ylabel('Pressure ratio [-]')           
plt.savefig('pressure_ratio.pdf')
plt.show()
plt.close()

##################################################

x = np.array(df[1:]['m_dot_ref'], dtype=float) * 0.000125998 #convert from lbm/hr to kg/s
y = np.array(df[1:]['m_dot_inj'], dtype=float) * 0.000125998 #convert from lbm/hr to kg/s
c = np.array(df[1:]['PR'], dtype=float)  # color of points (color bar points)
s = 60  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)

# Add a colorbar
cbar = plt.colorbar(im, ax=ax)

# set the color limits
im.set_clim(1000, 2600)
cbar.ax.set_ylabel('Discharge Pressure [kPa]')

#plt.ylim(0,22.5)
#plt.xlim(0,24.875)
plt.xlabel('Suction pressure [kPa]')
plt.ylabel('Pressure ratio [-]')
              
plt.savefig('injection_flow.pdf')
plt.show()
plt.close()
'''
Created on Sep 14, 2016

@author: ammarbahman

Note: this plots for Volumetric efficiency

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

#import the excel file
df = pd.read_excel("results.xlsx")

################################################################################
# volumetric efficiency /pressure ratio / injection temp
################################################################################
#assign axes
y = np.array(df[1:]['eta_vol'], dtype=float)
x = np.array(df[1:]['PR'], dtype=float)
c = (np.array(df[1:]['TC16_T_comp_inj'], dtype=float) + 459.67) * 5.0/9.0
s = 20  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(1000, 2600)
cbar.ax.set_ylabel('Injection temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(80,100)
#plt.xlim(0,24.875)
plt.ylabel('$\eta_v$ [\%]')
plt.xlabel('Pressure ratio [-]')           
plt.savefig('volumetric efficiency-pressure ratio-injection temp.pdf')
#plt.show()
plt.close()


################################################################################
# volumetric efficiency /pressure ratio / injection superheat
################################################################################
#assign axes
y = np.array(df[1:]['eta_vol'], dtype=float)
x = np.array(df[1:]['PR'], dtype=float)
c = np.array(df[1:]['DELTAT_sh_inj'], dtype=float) * 0.555556  # color of points (color bar points)
s = 20  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(1000, 2600)
cbar.ax.set_ylabel('Injection superheat [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(80,100)
#plt.xlim(0,24.875)
plt.ylabel('$\eta_v$ [\%]')
plt.xlabel('Pressure ratio [-]')           
plt.savefig('volumetric efficiency-pressure ratio-injection superheat.pdf')
#plt.show()
plt.close()


################################################################################
# volumetric efficiency /pressure ratio / injection mass flow rate
################################################################################
#assign axes
y = np.array(df[1:]['eta_vol'], dtype=float)
x = np.array(df[1:]['PR'], dtype=float)
c = np.array(df[1:]['m_dot_inj'], dtype=float) * 0.45359237 # color of points (color bar points)
s = 20  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(1000, 2600)
cbar.ax.set_ylabel('Injection mass flow rate [kg/hr]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(80,100)
#plt.xlim(0,24.875)
plt.ylabel('$\eta_v$ [\%]')
plt.xlabel('Pressure ratio [-]')           
plt.savefig('volumetric efficiency-pressure ratio-injection mass flow rate.pdf')
#plt.show()
plt.close()


################################################################################
# volumetric efficiency /pressure ratio / Norm injection mass flow rate
################################################################################
#assign axes
y = np.array(df[1:]['eta_vol'], dtype=float)
x = np.array(df[1:]['PR'], dtype=float)
c = np.array(df[1:]['NormalizedInjectionMassFlow'], dtype=float) * 100 # color of points (color bar points)
s = 20  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(0, 30)
cbar.ax.set_ylabel('Norm. inject. mass flow rate [\%]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(80,100)
#plt.xlim(0,24.875)
plt.ylabel('$\eta_v$ [\%]')
plt.xlabel('Pressure ratio [-]')           
plt.savefig('volumetric efficiency-pressure ratio-norm injection mass flow rate.pdf')
plt.show()
plt.close()


################################################################################
# volumetric efficiency /suction temperature / injection mass flow rate
################################################################################
#assign axes
y = np.array(df[1:]['eta_vol'], dtype=float)
x = (np.array(df[1:]['TC2_T_comp_suc'], dtype=float)  + 459.67) * 5.0/9.0
c = np.array(df[1:]['m_dot_inj'], dtype=float) * 0.45359237 # color of points (color bar points)
s = 20  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(1000, 2600)
cbar.ax.set_ylabel('Injection mass flow rate [kg/hr]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(80,100)
#plt.xlim(0,24.875)
plt.ylabel('$\eta_v$ [\%]')
plt.xlabel('Suction temperature [K]')           
plt.savefig('volumetric efficiency-suction temperature-injection mass flow rate.pdf')
#plt.show()
plt.close()


################################################################################
# volumetric efficiency /suction temperature / Norm injection mass flow rate
################################################################################
#assign axes
y = np.array(df[1:]['eta_vol'], dtype=float)
x = (np.array(df[1:]['TC2_T_comp_suc'], dtype=float)  + 459.67) * 5.0/9.0
c = np.array(df[1:]['NormalizedInjectionMassFlow'], dtype=float) * 100 # color of points (color bar points)
s = 20  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(0, 30)
cbar.ax.set_ylabel('Norm. inject. mass flow rate [\%]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(80,100)
#plt.xlim(0,24.875)
plt.ylabel('$\eta_v$ [\%]')
plt.xlabel('Suction temperature [K]')           
plt.savefig('volumetric efficiency-suction temperature-norm injection mass flow rate.pdf')
plt.show()
plt.close()
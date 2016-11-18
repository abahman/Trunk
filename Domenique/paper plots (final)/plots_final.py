'''
Created on Nov 18, 2016

@author: ammarbahman

Note: this plots for Discharge temperature

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
df = pd.read_excel("data_final.xlsx")

################################################################################
# isen_effic / T_inj_sh
################################################################################
#assign axes
y = np.array(df[1:]['Actual Isentropic Efficiency'], dtype=float) * 100
x = np.array(df[1:]['Injection Superheat (K)'], dtype=float)
#c = np.array(df[1:]['DELTAT_sh_inj'], dtype=float) * 5.0/9.0
#s = 2*c  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y)#, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
#cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(0, 32)
#cbar.ax.set_ylabel('Injection supeheat temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(30,100)
#plt.xlim(0,24.875)
plt.ylabel('Isentropic efficiency [\%]')
plt.xlabel('Injection superheat temperature [K]')           
plt.savefig('isentropic_Tinj.pdf')
#plt.show()
plt.close()


# ################################################################################
# # volumetric efficiency / Norm injection mass flow rate / discharge temp
# ################################################################################
# #assign axes
# y = np.array(df[1:]['eta_vol'], dtype=float)
# x = np.array(df[1:]['NormalizedInjectionMassFlow'], dtype=float) * 100
# c = (np.array(df[1:]['TC3_T_comp_dis'], dtype=float) + 459.67) * 5.0/9.0
# s = 20  # size of points
# 
# fig, ax = plt.subplots()
# im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# # Add a colorbar
# cbar = plt.colorbar(im, ax=ax)
# # set the color limits
# #im.set_clim(1000, 2600)
# cbar.ax.set_ylabel('Discharge temperature [K]')
# #ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
# plt.ylim(80,100)
# plt.xlim(0,30)
# plt.ylabel('$\eta_v$ [\%]')
# plt.xlabel('Norm. inject. mass flow rate [\%]')           
# plt.savefig('volumetric efficiency-norm injection mass flow rate-discharge temp.pdf')
# plt.show()
# plt.close()
#
#
################################################################################
# vol_effic / T_inj_sh
################################################################################
#assign axes
y = np.array(df[1:]['Actual Volumetric Efficiency'], dtype=float) * 100
x = np.array(df[1:]['Injection Superheat (K)'], dtype=float)
#c = np.array(df[1:]['DELTAT_sh_inj'], dtype=float) * 5.0/9.0
#s = 2*c  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y)#, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
#cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(0, 32)
#cbar.ax.set_ylabel('Injection supeheat temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(80,100)
#plt.xlim(0,24.875)
plt.ylabel('Volumetric efficiency [\%]')
plt.xlabel('Injection superheat temperature [K]')           
plt.savefig('volumetric_Tinj.pdf')
#plt.show()
plt.close()


################################################################################
# T_dis / T_inj_sh
################################################################################
#assign axes
y = np.array(df[1:]['Actual Discharge Temperature (K)'], dtype=float)
x = np.array(df[1:]['Injection Superheat (K)'], dtype=float)
#c = np.array(df[1:]['DELTAT_sh_inj'], dtype=float) * 5.0/9.0
#s = 2*c  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y)#, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
#cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(0, 32)
#cbar.ax.set_ylabel('Injection supeheat temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
#plt.ylim(80,100)
#plt.xlim(0,24.875)
plt.ylabel('Discharge tempearture [K]')
plt.xlabel('Injection superheat temperature [K]')           
plt.savefig('Tdis_Tinj.pdf')
#plt.show()
plt.close()


################################################################################
# m_inj_ratio / T_inj_sh
################################################################################
#assign axes
y = np.array(df[1:]['Actual Injection Ratio'], dtype=float) * 100
x = np.array(df[1:]['Injection Superheat (K)'], dtype=float)
#c = np.array(df[1:]['DELTAT_sh_inj'], dtype=float) * 5.0/9.0
#s = 2*c  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y)#, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
#cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(0, 32)
#cbar.ax.set_ylabel('Injection supeheat temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
#plt.ylim(80,100)
#plt.xlim(0,24.875)
plt.ylabel('$\dot m_{inj}/\dot m_{tot}$ [\%]')
plt.xlabel('Injection superheat temperature [K]')           
plt.savefig('minj_Tinj.pdf')
#plt.show()
plt.close()


################################################################################
# f_loss / T_inj_sh
################################################################################
# #assign axes
# y = np.array(df[1:][''], dtype=float)
# x = np.array(df[1:]['Injection Superheat (K)'], dtype=float)
# #c = np.array(df[1:]['DELTAT_sh_inj'], dtype=float) * 5.0/9.0
# #s = 2*c  # size of points
# 
# fig, ax = plt.subplots()
# im = ax.scatter(x, y)#, c=c, s=s, cmap=plt.cm.jet)
# # Add a colorbar
# #cbar = plt.colorbar(im, ax=ax)
# # set the color limits
# #im.set_clim(0, 32)
# #cbar.ax.set_ylabel('Injection supeheat temperature [K]')
# #ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
# #plt.ylim(80,100)
# #plt.xlim(0,24.875)
# plt.ylabel('$f_{loss}$ [\%]')
# plt.xlabel('Injection superheat temperature [K]')           
# plt.savefig('floss_Tinj.pdf')
# plt.show()
# plt.close()


################################################################################
# isen_effic / T_inj_sh /T_cond / T_evap 
################################################################################
#assign axes
y = np.array(df[1:]['T_cond [K]'], dtype=float)
x = np.array(df[1:]['T_evap [K]'], dtype=float)
c = np.array(df[1:]['Actual Isentropic Efficiency'], dtype=float) * 100
s = np.array(df[1:]['Injection Superheat (K)'], dtype=float)

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(60, 68)
cbar.ax.set_ylabel('Isentropic efficiency [\%]')
ax.text(0.7,0.05,'Markersize (injection superheat) {:0.0g}'.format(min(s))+'$-${:0.0f} K'.format(max(s)),ha='center',va='center',transform = ax.transAxes,fontsize = 7)
#plt.ylim(30,100)
#plt.xlim(0,24.875)
plt.ylabel('Condensation tempearture [K]')
plt.xlabel('Evaporation temperature [K]')           
plt.savefig('isentropic_Tinj_Tcond_Tevap.pdf')
#plt.show()
plt.close()


################################################################################
# vol_effic / T_inj_sh /T_cond / T_evap 
################################################################################
#assign axes
y = np.array(df[1:]['T_cond [K]'], dtype=float)
x = np.array(df[1:]['T_evap [K]'], dtype=float)
c = np.array(df[1:]['Actual Volumetric Efficiency'], dtype=float) * 100
s = np.array(df[1:]['Injection Superheat (K)'], dtype=float)
 
fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(84, 92)
cbar.ax.set_ylabel('Volumetric efficiency [\%]')
ax.text(0.7,0.05,'Markersize (injection superheat) {:0.0g}'.format(min(s))+'$-${:0.0f} K'.format(max(s)),ha='center',va='center',transform = ax.transAxes,fontsize = 7)
#plt.ylim(30,100)
#plt.xlim(0,24.875)
plt.ylabel('Condensation tempearture [K]')
plt.xlabel('Evaporation temperature [K]')           
plt.savefig('volumetric_Tinj_Tcond_Tevap.pdf')
#plt.show()
plt.close()


################################################################################
# T_dis / T_inj_sh /T_cond / T_evap 
################################################################################
#assign axes
y = np.array(df[1:]['T_cond [K]'], dtype=float)
x = np.array(df[1:]['T_evap [K]'], dtype=float)
c = np.array(df[1:]['Actual Discharge Temperature (K)'], dtype=float)
s = np.array(df[1:]['Injection Superheat (K)'], dtype=float)
 
fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(340, 380)
cbar.ax.set_ylabel('Discharge temperature [K]')
ax.text(0.7,0.05,'Markersize (injection superheat) {:0.0g}'.format(min(s))+'$-${:0.0f} K'.format(max(s)),ha='center',va='center',transform = ax.transAxes,fontsize = 7)
#plt.ylim(30,100)
#plt.xlim(0,24.875)
plt.ylabel('Condensation tempearture [K]')
plt.xlabel('Evaporation temperature [K]')           
plt.savefig('T_dis_Tinj_Tcond_Tevap.pdf')
#plt.show()
plt.close()


################################################################################
# m_ratio / T_inj_sh /T_cond / T_evap 
################################################################################
#assign axes
y = np.array(df[1:]['T_cond [K]'], dtype=float)
x = np.array(df[1:]['T_evap [K]'], dtype=float)
c = np.array(df[1:]['Actual Injection Ratio'], dtype=float) * 100
s = np.array(df[1:]['Injection Superheat (K)'], dtype=float)
 
fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(0, 30)
cbar.ax.set_ylabel('$\dot m_{inj}/\dot m_{tot}$ [\%]')
ax.text(0.7,0.05,'Markersize (injection superheat) {:0.0g}'.format(min(s))+'$-${:0.0f} K'.format(max(s)),ha='center',va='center',transform = ax.transAxes,fontsize = 7)
#plt.ylim(30,100)
#plt.xlim(0,24.875)
plt.ylabel('Condensation tempearture [K]')
plt.xlabel('Evaporation temperature [K]')           
plt.savefig('minj_Tinj_Tcond_Tevap.pdf')
#plt.show()
plt.close()


################################################################################
# f_loss / T_inj_sh /T_cond / T_evap 
################################################################################
# #assign axes
# y = np.array(df[1:]['T_cond [K]'], dtype=float)
# x = np.array(df[1:]['T_evap [K]'], dtype=float)
# c = np.array(df[1:][''], dtype=float)
# s = np.array(df[1:]['Injection Superheat (K)'], dtype=float)
#  
# fig, ax = plt.subplots()
# im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# # Add a colorbar
# cbar = plt.colorbar(im, ax=ax)
# # set the color limits
# im.set_clim(0, 30)
# cbar.ax.set_ylabel('$f_{loss}$ [\%]')
# ax.text(0.7,0.05,'Markersize (injection superheat) {:0.0g}'.format(min(s))+'$-${:0.0f} K'.format(max(s)),ha='center',va='center',transform = ax.transAxes,fontsize = 7)
# #plt.ylim(30,100)
# #plt.xlim(0,24.875)
# plt.ylabel('Condensation tempearture [K]')
# plt.xlabel('Evaporation temperature [K]')           
# plt.savefig('floss_Tinj_Tcond_Tevap.pdf')
# plt.show()
# plt.close()
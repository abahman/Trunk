'''
Created on Sep 13, 2016

@author: ammarbahman

Note: this plots for VI compressor testing in calorimeter paper

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
# pressure ratio vs suction pressure
################################################################################
#assign axes
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
ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
#plt.ylim(0,22.5)
#plt.xlim(0,24.875)
plt.xlabel('Suction pressure [kPa]')
plt.ylabel('Pressure ratio [-]')           
plt.savefig('pressure_ratio_vs_suction_pressure.pdf')
#plt.show()
plt.close()


################################################################################
# Norm suction mass vs Norm injection mass
################################################################################
#assign axes
x = np.array(df[1:]['m_dot_ref']/df[1:]['m_dot_total'], dtype=float) * 100 #normalized flow rate
y = np.array(df[1:]['m_dot_inj']/df[1:]['m_dot_total'], dtype=float) * 100 #normalized flow rate
c = np.array(df[1:]['PR'], dtype=float)  # color of points (color bar points)
s = np.array(df[1:]['PE2_p_comp_dis'], dtype=float)  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(2, 7)
cbar.ax.set_ylabel('Pressure ratio [-]')
ax.text(0.75,0.95,'Markersize ($P_{dis}$)'+' {:0.0f} to '.format(np.min(s)) +'{:0.0f} psia'.format(np.max(s)),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(0,30)
plt.xlim(70,100)
plt.xlabel('Norm. suction mass flow rate [\%]')
plt.ylabel('Norm. injection mass flow rate [\%]')            
plt.savefig('Norm_suction_mass_vs_Norm_injection_mass.pdf')
#plt.show()
plt.close()


################################################################################
# suction superheat vs suction pressure
################################################################################
#assign axes
x = np.array(df[1:]['PE1_p_comp_suc'], dtype=float) * 6.89476 #convert from psia to kPa
y = np.array(df[1:]['DELTAT_sh_suc'], dtype=float) * 0.555556 #convert delta R to delta K
c = np.array(df[1:]['PR'], dtype=float)  # color of points (color bar points)
s = np.array(df[1:]['PE2_p_comp_dis'], dtype=float)  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(2, 7)
cbar.ax.set_ylabel('Pressure ratio[-]')
ax.text(0.75,0.95,'Markersize ($P_{dis}$)'+' {:0.0f} to '.format(np.min(s)) +'{:0.0f} psia'.format(np.max(s)),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(0,20)
#plt.xlim(0,24.875)
plt.xlabel('Suction pressure [kPa]')
plt.ylabel('Suction superheat [K]')           
plt.savefig('suction_superheat_vs_suction_pressure.pdf')
#plt.show()
plt.close()


################################################################################
# suction superheat vs discharge pressure
################################################################################
#assign axes
x = np.array(df[1:]['PE2_p_comp_dis'], dtype=float) * 6.89476 #convert from psia to kPa
y = np.array(df[1:]['DELTAT_sh_suc'], dtype=float) * 0.555556 #convert delta R to delta K
c = np.array(df[1:]['PR'], dtype=float)  # color of points (color bar points)
s = 60  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(2, 7)
cbar.ax.set_ylabel('Pressure ratio[-]')
ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(0,20)
#plt.xlim(0,24.875)
plt.xlabel('Discharge pressure [kPa]')
plt.ylabel('Suction superheat [K]')           
plt.savefig('suction_superheat_vs_discharge_pressure.pdf')
plt.show()
plt.close()


################################################################################
# injection superheat vs suction superheat
################################################################################
#assign axes
x = np.array(df[1:]['DELTAT_sh_suc'], dtype=float) * 0.555556 #convert delta R to delta K
y = np.array(df[1:]['DELTAT_sh_inj'], dtype=float) * 0.555556 #convert delta R to delta K
c = np.array(df[1:]['PR'], dtype=float)  # color of points (color bar points)
s = np.array(df[1:]['PE2_p_comp_dis'], dtype=float)  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(2, 7)
cbar.ax.set_ylabel('Pressure ratio[-]')
ax.text(0.75,0.95,'Markersize ($P_{dis}$)'+' {:0.0f} to '.format(np.min(s)) +'{:0.0f} psia'.format(np.max(s)),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(0,40)
plt.xlim(0,20)
plt.xlabel('Suction superheat [K]')
plt.ylabel('Injection superheat [K]')           
plt.savefig('injection_superheat_vs_suction_superheat.pdf')
#plt.show()
plt.close()

################################################################################
# volumetric eff vs pressure ratio
################################################################################
#assign axes
x = np.array(df[1:]['PR'], dtype=float)
y = np.array(df[1:]['eta_vol'], dtype=float)
c = (np.array(df[1:]['TC3_T_comp_dis'], dtype=float) + 459.67) * 5.0/9.0 #convert F to  K# color of points (color bar points)
s = np.array(df[1:]['PE2_p_comp_dis'], dtype=float)  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(335, 385)
cbar.ax.set_ylabel('Discharge temperature [K]')
ax.text(0.75,0.95,'Markersize ($P_{dis}$)'+' {:0.0f} to '.format(np.min(s)) +'{:0.0f} psia'.format(np.max(s)),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(80,100)
plt.xlim(2,7)
plt.xlabel('Pressure ratio [-]')
plt.ylabel('Volumetric efficiency [\%]')           
plt.savefig('volume_eff_vs_PR.pdf')
#plt.show()
plt.close()


################################################################################
# heat loss vs discharge temp
################################################################################
#assign axes
x = (np.array(df[1:]['TC3_T_comp_dis'], dtype=float) + 459.67) * 5.0/9.0 #convert F to  K
y = np.array(df[1:]['f_Q'], dtype=float)
c = (np.array(df[1:]['T_evap_stpt'], dtype=float) + 459.67) * 5.0/9.0 # color of points (color bar points)
s = np.array(df[1:]['PE2_p_comp_dis'], dtype=float)  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(260, 280)
cbar.ax.set_ylabel('Saturated suction temperature [K]')
ax.text(0.75,0.95,'Markersize ($P_{dis}$)'+' {:0.0f} to '.format(np.min(s)) +'{:0.0f} psia'.format(np.max(s)),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
#plt.ylim(0,40)
#plt.xlim(0,20)
plt.xlabel('Discharge temperature [K]')
plt.ylabel('$f_Q$ [\%]')           
plt.savefig('heat_loss_vs_discharge_temp.pdf')
#plt.show()
plt.close()


################################################################################
# isentropic eff vs volume ratio
################################################################################
#assign axes
x = np.array(df[1:]['VolumeRatio'], dtype=float) 
y = np.array(df[1:]['eta_isen'], dtype=float)
c = np.array(df[1:]['PR'], dtype=float) # color of points (color bar points)
s = 60  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(2, 7)
cbar.ax.set_ylabel('Pressure ratio [-]')
ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(60,70)
plt.xlim(2,6)
plt.xlabel('Volume ratio ($v_{suc}/v_{dis}$) [-]')
plt.ylabel('Overall isentropic efficiency [\%]')           
plt.savefig('isentropic_eff_vs_volume_ratio.pdf')
#plt.show()
plt.close()


################################################################################
# heat loss vs ambient temp 1
################################################################################
#assign axes
x = (np.array(df[1:]['TC12_T_chamb_amb'], dtype=float) + 459.67) * 5.0/9.0 #convert F to K
y = np.array(df[1:]['Q_dot_loss'], dtype=float) * 0.29307107 # convert btu/hr to Watts
c = (np.array(df[1:]['T_evap_stpt'], dtype=float) + 459.67) * 5.0/9.0 # color of points (color bar points)
s = 60  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(260, 280)
cbar.ax.set_ylabel('Saturated suction temperature [K]')
ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
#plt.ylim(60,70)
plt.xlim(300,316)
plt.xlabel('Ambient temperature [K]')
plt.ylabel('Heat loss [W]')           
plt.savefig('heat_loss_vs_ambient_temp_1.pdf')
#plt.show()
plt.close()


################################################################################
# heat loss vs ambient temp 2
################################################################################
#assign axes
x = (np.array(df[1:]['TC12_T_chamb_amb'], dtype=float) + 459.67) * 5.0/9.0 #convert F to K
y = np.array(df[1:]['f_Q'], dtype=float)
c = (np.array(df[1:]['T_evap_stpt'], dtype=float) + 459.67) * 5.0/9.0 # color of points (color bar points)
s = 60  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(260, 280)
cbar.ax.set_ylabel('Saturated suction temperature [K]')
ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
#plt.ylim(60,70)
plt.xlim(300,316)
plt.xlabel('Ambient temperature [K]')
plt.ylabel('$f_Q$ [\%]')           
plt.savefig('heat_loss_vs_ambient_temp_2.pdf')
#plt.show()
plt.close()


################################################################################
# discharge volume ratio eff vs pressure ratio
################################################################################
#assign axes
x = np.array(df[1:]['PR'], dtype=float)
y = np.array(df[1:]['DVR'], dtype=float)
c = np.array(df[1:]['NormalizedInjectionMassFlow'], dtype=float) * 100 # color of points (color bar points)
s = np.array(df[1:]['PE2_p_comp_dis'], dtype=float)  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(0, 30)
cbar.ax.set_ylabel('Normalized injection mass flow rate [\%]')
ax.text(0.75,0.95,'Markersize ($P_{dis}$)'+' {:0.0f} to '.format(np.min(s)) +'{:0.0f} psia'.format(np.max(s)),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
#plt.ylim(60,70)
#plt.xlim(300,316)
l = plt.axhline(y=100, color='b') #draw a blueline at y=100%

bbox_props = dict(boxstyle="rarrow", fc=(0.8,0.9,0.9), ec="b", lw=1)
t = ax.text(3, 100, "under-\n compression", ha="center", va="center", rotation=-90,
            size=8,
            bbox=bbox_props)

bb = t.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.6)

#ax.text(0.2,0.2,'undercompression',ha='center',va='center',transform = ax.transAxes,fontsize = 8) #text
plt.xlabel('Pressure ratio [-]')
plt.ylabel('Discharge pocket volume ratio [\%]')           
plt.savefig('discharge_vol_eff_vs_PR.pdf')
#plt.show()
plt.close()


################################################################################
# isentropic eff vs pressure ratio
################################################################################
#assign axes
x = np.array(df[1:]['PR'], dtype=float) 
y = np.array(df[1:]['eta_isen'], dtype=float)
c = np.array(df[1:]['NormalizedInjectionMassFlow'], dtype=float) * 100 # color of points (color bar points)
s = np.array(df[1:]['PE2_p_comp_dis'], dtype=float)  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(0, 30)
cbar.ax.set_ylabel('Normalized injection mass flow rate [\%]')
ax.text(0.75,0.95,'Markersize ($P_{dis}$)'+' {:0.0f} to '.format(np.min(s)) +'{:0.0f} psia'.format(np.max(s)),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(60,70)
plt.xlim(2,7)
plt.xlabel('Pressure ratio [-]')
plt.ylabel('Overall isentropic efficiency [\%]')           
plt.savefig('isentropic_eff_vs_PR.pdf')
#plt.show()
plt.close()


################################################################################
# Norm injection mass flow vs pressure ratio
################################################################################
#assign axes
x = np.array(df[1:]['PR'], dtype=float) 
y = np.array(df[1:]['NormalizedInjectionMassFlow'], dtype=float) * 100
c = (np.array(df[1:]['TC3_T_comp_dis'], dtype=float) + 459.67) * 5.0/9.0  # color of points (color bar points)
s = np.array(df[1:]['PE2_p_comp_dis'], dtype=float)  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(340, 385)
cbar.ax.set_ylabel('Discharge temperature [K]')
ax.text(0.75,0.95,'Markersize ($P_{dis}$)'+' {:0.0f} to '.format(np.min(s)) +'{:0.0f} psia'.format(np.max(s)),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(0,35)
plt.xlim(2,7)
plt.xlabel('Pressure ratio [-]')
plt.ylabel('Normalized injection mass flow rate [\%]')           
plt.savefig('Norm_injection_mass_vs_PR.pdf')
#plt.show()
plt.close()
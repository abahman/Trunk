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
# isen_effic / PR 
################################################################################
#assign axes
y = np.array(df[0:11]['Actual Isentropic Efficiency'], dtype=float) * 100
x = np.array(df[0:11]['PressureRatio'], dtype=float)
y1 = np.array(df[11:]['Actual Isentropic Efficiency'], dtype=float) * 100
x1 = np.array(df[11:]['PressureRatio'], dtype=float)
#c = np.array(df[1:]['DELTAT_sh_inj'], dtype=float) * 5.0/9.0
#s = 2*c  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y,lw=0.2,label ='Wet injection',alpha=0.9)#, c=c, s=s, cmap=plt.cm.jet)
im = ax.scatter(x1, y1, c='red', marker='s',lw=0.2,label ='Vapor injection',alpha=0.9)
# Add a colorbar
#cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(0, 32)
#cbar.ax.set_ylabel('Injection supeheat temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(50,80)
#plt.xlim(0,24.875)
plt.ylabel('Isentropic efficiency [\%]')
plt.xlabel('Pressure ratio [-]')
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()
plt.savefig('isentropic_PR.pdf')
#plt.show()
plt.close()


################################################################################
# vol_effic / PR
################################################################################
#assign axes
y = np.array(df[0:11]['Actual Volumetric Efficiency'], dtype=float) * 100
x = np.array(df[0:11]['PressureRatio'], dtype=float)
y1 = np.array(df[11:]['Actual Volumetric Efficiency'], dtype=float) * 100
x1 = np.array(df[11:]['PressureRatio'], dtype=float)
#c = np.array(df[0:]['DELTAT_sh_inj'], dtype=float) * 5.0/9.0
#s = 2*c  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y,lw=0.2,label ='wet injection',alpha=0.9)#, c=c, s=s, cmap=plt.cm.jet)
im = ax.scatter(x1, y1, c='red', marker='s',lw=0.2,label ='Vapor injection',alpha=0.9)
# Add a colorbar
#cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(0, 32)
#cbar.ax.set_ylabel('Injection supeheat temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(80,100)
#plt.xlim(0,24.875)
plt.ylabel('Volumetric efficiency [\%]')
plt.xlabel('Pressure ratio [-]') 
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()           
plt.savefig('volumetric_PR.pdf')
#plt.show()
plt.close()


################################################################################
# T_dis / PR
################################################################################
#assign axes
y = np.array(df[0:11]['Actual Discharge Temperature (K)'], dtype=float)
x = np.array(df[0:11]['PressureRatio'], dtype=float)
y1 = np.array(df[11:]['Actual Discharge Temperature (K)'], dtype=float)
x1 = np.array(df[11:]['PressureRatio'], dtype=float)
#c = np.array(df[0:]['DELTAT_sh_inj'], dtype=float) * 5.0/9.0
#s = 2*c  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y,lw=0.2,label ='wet injection',alpha=0.9)#, c=c, s=s, cmap=plt.cm.jet)
im = ax.scatter(x1, y1, c='red', marker='s',lw=0.2,label ='Vapor injection',alpha=0.9)
# Add a colorbar
#cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(0, 32)
#cbar.ax.set_ylabel('Injection supeheat temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
#plt.ylim(80,100)
#plt.xlim(0,24.875)
plt.ylabel('Discharge tempearture [K]')
plt.xlabel('Pressure ratio [-]')
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()  
plt.savefig('Tdis_PR.pdf')
#plt.show()
plt.close()


################################################################################
# m_inj_ratio / PR
################################################################################
#assign axes
y = np.array(df[0:11]['Actual Injection Ratio'], dtype=float) * 100
x = np.array(df[0:11]['PressureRatio'], dtype=float)
y1 = np.array(df[11:]['Actual Injection Ratio'], dtype=float) * 100
x1 = np.array(df[11:]['PressureRatio'], dtype=float)
#c = np.array(df[1:]['DELTAT_sh_inj'], dtype=float) * 5.0/9.0
#s = 2*c  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y,lw=0.2,label ='wet injection',alpha=0.9)#, c=c, s=s, cmap=plt.cm.jet)
im = ax.scatter(x1, y1, c='red', marker='s',lw=0.2,label ='Vapor injection',alpha=0.9)
# Add a colorbar
#cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(0, 32)
#cbar.ax.set_ylabel('Injection supeheat temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(0,40)
#plt.xlim(0,24.875)
plt.ylabel('$\dot m_{inj}/\dot m_{suc}$ [\%]')
plt.xlabel('Pressure ratio [-]')
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()        
plt.savefig('minj_PR.pdf')
#plt.show()
plt.close()


################################################################################
# f_loss / PR
################################################################################
#assign axes
y = np.array(df[0:11]['actual heat loss'], dtype=float)
x = np.array(df[0:11]['PressureRatio'], dtype=float)
y1 = np.array(df[11:]['actual heat loss'], dtype=float)
x1 = np.array(df[11:]['PressureRatio'], dtype=float)
#c = np.array(df[1:]['DELTAT_sh_inj'], dtype=float) * 5.0/9.0
#s = 2*c  # size of points
  
fig, ax = plt.subplots()
im = ax.scatter(x, y,lw=0.2,label ='wet injection',alpha=0.9)#, c=c, s=s, cmap=plt.cm.jet)
im = ax.scatter(x1, y1, c='red', marker='s',lw=0.2,label ='Vapor injection',alpha=0.9)
# Add a colorbar
#cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(0, 32)
#cbar.ax.set_ylabel('Injection supeheat temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(0,10)
#plt.xlim(0,24.875)
plt.ylabel('$f_{loss}$ [\%]')
plt.xlabel('Pressure ratio [-]')   
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()      
plt.savefig('floss_PR.pdf')
#plt.show()
plt.close()


################################################################################
# isen_effic / T_inj_sh
################################################################################
#assign axes
y = np.array(df[0:11]['Actual Isentropic Efficiency'], dtype=float) * 100
x = np.array(df[0:11]['Injection Superheat (K)'], dtype=float)
y1 = np.array(df[11:]['Actual Isentropic Efficiency'], dtype=float) * 100
x1 = np.array(df[11:]['Injection Superheat (K)'], dtype=float)
#c = np.array(df[1:]['DELTAT_sh_inj'], dtype=float) * 5.0/9.0
#s = 2*c  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y,lw=0.2,label ='Wet injection',alpha=0.9)#, c=c, s=s, cmap=plt.cm.jet)
im = ax.scatter(x1, y1, c='red', marker='s',lw=0.2,label ='Vapor injection',alpha=0.9)
# Add a colorbar
#cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(0, 32)
#cbar.ax.set_ylabel('Injection supeheat temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(50,80)
#plt.xlim(0,24.875)
plt.ylabel('Isentropic efficiency [\%]')
plt.xlabel('Injection superheat temperature [K]')
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()
plt.savefig('isentropic_Tinj.pdf')
#plt.show()
plt.close()



################################################################################
# vol_effic / T_inj_sh
################################################################################
#assign axes
y = np.array(df[0:11]['Actual Volumetric Efficiency'], dtype=float) * 100
x = np.array(df[0:11]['Injection Superheat (K)'], dtype=float)
y1 = np.array(df[11:]['Actual Volumetric Efficiency'], dtype=float) * 100
x1 = np.array(df[11:]['Injection Superheat (K)'], dtype=float)
#c = np.array(df[0:]['DELTAT_sh_inj'], dtype=float) * 5.0/9.0
#s = 2*c  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y,lw=0.2,label ='wet injection',alpha=0.9)#, c=c, s=s, cmap=plt.cm.jet)
im = ax.scatter(x1, y1, c='red', marker='s',lw=0.2,label ='Vapor injection',alpha=0.9)
# Add a colorbar
#cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(0, 32)
#cbar.ax.set_ylabel('Injection supeheat temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(80,100)
#plt.xlim(0,24.875)
plt.ylabel('Volumetric efficiency [\%]')
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()    
plt.xlabel('Injection superheat temperature [K]')           
plt.savefig('volumetric_Tinj.pdf')
#plt.show()
plt.close()


################################################################################
# T_dis / T_inj_sh
################################################################################
#assign axes
y = np.array(df[0:11]['Actual Discharge Temperature (K)'], dtype=float)
x = np.array(df[0:11]['Injection Superheat (K)'], dtype=float)
y1 = np.array(df[11:]['Actual Discharge Temperature (K)'], dtype=float)
x1 = np.array(df[11:]['Injection Superheat (K)'], dtype=float)
#c = np.array(df[0:]['DELTAT_sh_inj'], dtype=float) * 5.0/9.0
#s = 2*c  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y,lw=0.2,label ='wet injection',alpha=0.9)#, c=c, s=s, cmap=plt.cm.jet)
im = ax.scatter(x1, y1, c='red', marker='s',lw=0.2,label ='Vapor injection',alpha=0.9)
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
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()      
plt.savefig('Tdis_Tinj.pdf')
#plt.show()
plt.close()


################################################################################
# m_inj_ratio / T_inj_sh
################################################################################
#assign axes
y = np.array(df[0:11]['Actual Injection Ratio'], dtype=float) * 100
x = np.array(df[0:11]['Injection Superheat (K)'], dtype=float)
y1 = np.array(df[11:]['Actual Injection Ratio'], dtype=float) * 100
x1 = np.array(df[11:]['Injection Superheat (K)'], dtype=float)
#c = np.array(df[1:]['DELTAT_sh_inj'], dtype=float) * 5.0/9.0
#s = 2*c  # size of points

fig, ax = plt.subplots()
im = ax.scatter(x, y,lw=0.2,label ='wet injection',alpha=0.9)#, c=c, s=s, cmap=plt.cm.jet)
im = ax.scatter(x1, y1, c='red', marker='s',lw=0.2,label ='Vapor injection',alpha=0.9)
# Add a colorbar
#cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(0, 32)
#cbar.ax.set_ylabel('Injection supeheat temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(0,40)
#plt.xlim(0,24.875)
plt.ylabel('$\dot m_{inj}/\dot m_{suc}$ [\%]')
plt.xlabel('Injection superheat temperature [K]')
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()         
plt.savefig('minj_Tinj.pdf')
#plt.show()
plt.close()


################################################################################
# f_loss / T_inj_sh
################################################################################
#assign axes
y = np.array(df[0:11]['actual heat loss'], dtype=float)
x = np.array(df[0:11]['Injection Superheat (K)'], dtype=float)
y1 = np.array(df[11:]['actual heat loss'], dtype=float)
x1 = np.array(df[11:]['Injection Superheat (K)'], dtype=float)
#c = np.array(df[1:]['DELTAT_sh_inj'], dtype=float) * 5.0/9.0
#s = 2*c  # size of points
  
fig, ax = plt.subplots()
im = ax.scatter(x, y,lw=0.2,label ='wet injection',alpha=0.9)#, c=c, s=s, cmap=plt.cm.jet)
im = ax.scatter(x1, y1, c='red', marker='s',lw=0.2,label ='Vapor injection',alpha=0.9)
# Add a colorbar
#cbar = plt.colorbar(im, ax=ax)
# set the color limits
#im.set_clim(0, 32)
#cbar.ax.set_ylabel('Injection supeheat temperature [K]')
#ax.text(0.8,0.95,'Markersize (speed) {:0.0f} Hz'.format(s),ha='center',va='center',transform = ax.transAxes,fontsize = 8)
plt.ylim(0,10)
#plt.xlim(0,24.875)
plt.ylabel('$f_{loss}$ [\%]')
plt.xlabel('Injection superheat temperature [K]')   
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()      
plt.savefig('floss_Tinj.pdf')
#plt.show()
plt.close()


################################################################################
# isen_effic / T_inj_sh /T_cond / T_evap 
################################################################################
#assign axes
y = np.array(df[0:11]['T_cond [K]'], dtype=float)
x = np.array(df[0:11]['T_evap [K]'], dtype=float)
c = np.array(df[0:11]['Actual Isentropic Efficiency'], dtype=float) * 100
s = np.array(df[0:11]['Injection Superheat (K)'], dtype=float)
y1 = np.array(df[11:]['T_cond [K]'], dtype=float)
x1 = np.array(df[11:]['T_evap [K]'], dtype=float)
c1 = np.array(df[11:]['Actual Isentropic Efficiency'], dtype=float) * 100
s1 = np.array(df[11:]['Injection Superheat (K)'], dtype=float)
 
fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, cmap=plt.cm.jet, marker = 'o', lw=0.2,label = 'Wet injection',alpha=0.9)
im = ax.scatter(x1, y1, c=c1, s=s1, cmap=plt.cm.jet, marker = 's', lw=0.2,label = 'Vapor injection',alpha=0.9)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(60, 68)
cbar.ax.set_ylabel('Isentropic efficiency [\%]')
#ax.text(0.68,0.05,'Markersize (injection superheat) ${:0.0g}$'.format(min(s))+' to {:0.0f} K'.format(max(s1)),ha='center',va='center',transform = ax.transAxes,fontsize = 7)
#plt.ylim(30,100)
#plt.xlim(0,24.875)
plt.ylabel('Condensation tempearture [K]')
plt.xlabel('Evaporation temperature [K]')
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()     
plt.savefig('isentropic_Tinj_Tcond_Tevap.pdf')
#plt.show()
plt.close()


################################################################################
# vol_effic / T_inj_sh /T_cond / T_evap 
################################################################################
#assign axes
y = np.array(df[0:11]['T_cond [K]'], dtype=float)
x = np.array(df[0:11]['T_evap [K]'], dtype=float)
c = np.array(df[0:11]['Actual Volumetric Efficiency'], dtype=float) * 100
s = np.array(df[0:11]['Injection Superheat (K)'], dtype=float)
y1 = np.array(df[11:]['T_cond [K]'], dtype=float)
x1 = np.array(df[11:]['T_evap [K]'], dtype=float)
c1 = np.array(df[11:]['Actual Volumetric Efficiency'], dtype=float) * 100
s1 = np.array(df[11:]['Injection Superheat (K)'], dtype=float)
  
fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, cmap=plt.cm.jet, marker = 'o', lw=0.2,label = 'Wet injection',alpha=0.9)
im = ax.scatter(x1, y1, c=c1, cmap=plt.cm.jet, marker = 's', lw=0.2,label = 'Vapor injection',alpha=0.9)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(84, 92)
cbar.ax.set_ylabel('Volumetric efficiency [\%]')
#ax.text(0.68,0.05,'Markersize (injection superheat) ${:0.0g}$'.format(min(s))+' to {:0.0f} K'.format(max(s1)),ha='center',va='center',transform = ax.transAxes,fontsize = 7)
#plt.ylim(30,100)
#plt.xlim(0,24.875)
plt.ylabel('Condensation tempearture [K]')
plt.xlabel('Evaporation temperature [K]')
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()     
plt.savefig('volumetric_Tinj_Tcond_Tevap.pdf')
#plt.show()
plt.close()
 
 
################################################################################
# T_dis / T_inj_sh /T_cond / T_evap 
################################################################################
#assign axes
y = np.array(df[0:11]['T_cond [K]'], dtype=float)
x = np.array(df[0:11]['T_evap [K]'], dtype=float)
c = np.array(df[0:11]['Actual Discharge Temperature (K)'], dtype=float)
s = np.array(df[0:11]['Injection Superheat (K)'], dtype=float)
y1 = np.array(df[11:]['T_cond [K]'], dtype=float)
x1 = np.array(df[11:]['T_evap [K]'], dtype=float)
c1 = np.array(df[11:]['Actual Discharge Temperature (K)'], dtype=float)
s1 = np.array(df[11:]['Injection Superheat (K)'], dtype=float)
  
fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, cmap=plt.cm.jet, marker = 'o', lw=0.2,label = 'Wet injection',alpha=0.9)
im = ax.scatter(x1, y1, c=c1, cmap=plt.cm.jet, marker = 's', lw=0.2,label = 'Vapor injection',alpha=0.9)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(340, 380)
cbar.ax.set_ylabel('Discharge temperature [K]')
#ax.text(0.68,0.05,'Markersize (injection superheat) ${:0.0g}$'.format(min(s))+' to {:0.0f} K'.format(max(s1)),ha='center',va='center',transform = ax.transAxes,fontsize = 7)
#plt.ylim(30,100)
#plt.xlim(0,24.875)
plt.ylabel('Condensation tempearture [K]')
plt.xlabel('Evaporation temperature [K]')
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()        
plt.savefig('T_dis_Tinj_Tcond_Tevap.pdf')
#plt.show()
plt.close()
 
 
################################################################################
# m_ratio / T_inj_sh /T_cond / T_evap 
################################################################################
#assign axes
y = np.array(df[0:11]['T_cond [K]'], dtype=float)
x = np.array(df[0:11]['T_evap [K]'], dtype=float)
c = np.array(df[0:11]['Actual Injection Ratio'], dtype=float) * 100
s = np.array(df[0:11]['Injection Superheat (K)'], dtype=float)
y1 = np.array(df[11:]['T_cond [K]'], dtype=float)
x1 = np.array(df[11:]['T_evap [K]'], dtype=float)
c1 = np.array(df[11:]['Actual Injection Ratio'], dtype=float) * 100
s1 = np.array(df[11:]['Injection Superheat (K)'], dtype=float)
  
fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, cmap=plt.cm.jet, marker = 'o', lw=0.2,label = 'Wet injection',alpha=0.9)
im = ax.scatter(x1, y1, c=c1, cmap=plt.cm.jet, marker = 's', lw=0.2,label = 'Vapor injection',alpha=0.9)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(0, 30)
cbar.ax.set_ylabel('$\dot m_{inj}/\dot m_{suc}$ [\%]')
#ax.text(0.68,0.05,'Markersize (injection superheat) ${:0.0g}$'.format(min(s))+' to {:0.0f} K'.format(max(s1)),ha='center',va='center',transform = ax.transAxes,fontsize = 7)
#plt.ylim(30,100)
#plt.xlim(0,24.875)
plt.ylabel('Condensation tempearture [K]')
plt.xlabel('Evaporation temperature [K]')
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()  
plt.savefig('minj_Tinj_Tcond_Tevap.pdf')
#plt.show()
plt.close()
 
 
################################################################################
# f_loss / T_inj_sh /T_cond / T_evap 
################################################################################
#assign axes
y = np.array(df[0:11]['T_cond [K]'], dtype=float)
x = np.array(df[0:11]['T_evap [K]'], dtype=float)
c = np.array(df[0:11]['actual heat loss'], dtype=float)
s = np.array(df[0:11]['Injection Superheat (K)'], dtype=float)
y1 = np.array(df[11:]['T_cond [K]'], dtype=float)
x1 = np.array(df[11:]['T_evap [K]'], dtype=float)
c1 = np.array(df[11:]['actual heat loss'], dtype=float)
s1 = np.array(df[11:]['Injection Superheat (K)'], dtype=float)
   
fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, cmap=plt.cm.jet, marker = 'o', lw=0.2,label = 'Wet injection',alpha=0.9)
im = ax.scatter(x1, y1, c=c1, cmap=plt.cm.jet, marker = 's', lw=0.2,label = 'Vapor injection',alpha=0.9)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(2, 7)
cbar.ax.set_ylabel('$f_{loss}$ [\%]')
#ax.text(0.68,0.05,'Markersize (injection superheat) ${:0.0g}$'.format(min(s))+' to {:0.0f} K'.format(max(s1)),ha='center',va='center',transform = ax.transAxes,fontsize = 7)
#plt.ylim(30,100)
#plt.xlim(0,24.875)
plt.ylabel('Condensation tempearture [K]')
plt.xlabel('Evaporation temperature [K]')
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()       
plt.savefig('floss_Tinj_Tcond_Tevap.pdf')
#plt.show()
plt.close()


################################################################################
# isen_effic / T_inj_sh /T_cond / T_evap 
################################################################################
#assign axes
y = np.array(df[0:22]['T_cond [K]'], dtype=float)
x = np.array(df[0:22]['T_evap [K]'], dtype=float)
c = np.array(df[0:22]['Actual Isentropic Efficiency'], dtype=float) * 100
s = np.array(df[0:22]['Injection Superheat (K)'], dtype=float)
y1 = np.array(df[22:]['T_cond [K]'], dtype=float)
x1 = np.array(df[22:]['T_evap [K]'], dtype=float)
c1 = np.array(df[22:]['Actual Isentropic Efficiency'], dtype=float) * 100
s1 = np.array(df[22:]['Injection Superheat (K)'], dtype=float)

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, cmap=plt.cm.jet, marker = '^', lw=0.2,label = 'Fixed injection',alpha=0.9)
im = ax.scatter(x1, y1, c=c1, s=s1, cmap=plt.cm.jet, marker = '*', lw=0.2,label = 'Variable injection',alpha=0.9)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(60, 68)
cbar.ax.set_ylabel('Isentropic efficiency [\%]')
#ax.text(0.68,0.05,'Markersize (injection superheat) ${:0.0g}$'.format(min(s))+' to {:0.0f} K'.format(max(s1)),ha='center',va='center',transform = ax.transAxes,fontsize = 7)
#plt.ylim(30,100)
#plt.xlim(0,24.875)
plt.ylabel('Condensation tempearture [K]')
plt.xlabel('Evaporation temperature [K]')
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()
plt.savefig('isentropic_Tinj_Tcond_Tevap_FV.pdf')
#plt.show()
plt.close()


################################################################################
# vol_effic / T_inj_sh /T_cond / T_evap 
################################################################################
#assign axes
y = np.array(df[0:22]['T_cond [K]'], dtype=float)
x = np.array(df[0:22]['T_evap [K]'], dtype=float)
c = np.array(df[0:22]['Actual Volumetric Efficiency'], dtype=float) * 100
s = np.array(df[0:22]['Injection Superheat (K)'], dtype=float)
y1 = np.array(df[22:]['T_cond [K]'], dtype=float)
x1 = np.array(df[22:]['T_evap [K]'], dtype=float)
c1 = np.array(df[22:]['Actual Volumetric Efficiency'], dtype=float) * 100
s1 = np.array(df[22:]['Injection Superheat (K)'], dtype=float)
 
fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, cmap=plt.cm.jet, marker = '^', lw=0.2,label = 'Fixed injection',alpha=0.9)
im = ax.scatter(x1, y1, c=c1, cmap=plt.cm.jet, marker = '*', lw=0.2,label = 'Variable injection',alpha=0.9)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(84, 92)
cbar.ax.set_ylabel('Volumetric efficiency [\%]')
#ax.text(0.68,0.05,'Markersize (injection superheat) ${:0.0g}$'.format(min(s))+' to {:0.0f} K'.format(max(s1)),ha='center',va='center',transform = ax.transAxes,fontsize = 7)
#plt.ylim(30,100)
#plt.xlim(0,24.875)
plt.ylabel('Condensation tempearture [K]')
plt.xlabel('Evaporation temperature [K]')
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)  
plt.tight_layout()     
plt.savefig('volumetric_Tinj_Tcond_Tevap_FV.pdf')
#plt.show()
plt.close()


################################################################################
# T_dis / T_inj_sh /T_cond / T_evap 
################################################################################
#assign axes
y = np.array(df[0:22]['T_cond [K]'], dtype=float)
x = np.array(df[0:22]['T_evap [K]'], dtype=float)
c = np.array(df[0:22]['Actual Discharge Temperature (K)'], dtype=float)
s = np.array(df[0:22]['Injection Superheat (K)'], dtype=float)
y1 = np.array(df[22:]['T_cond [K]'], dtype=float)
x1 = np.array(df[22:]['T_evap [K]'], dtype=float)
c1 = np.array(df[22:]['Actual Discharge Temperature (K)'], dtype=float)
s1 = np.array(df[22:]['Injection Superheat (K)'], dtype=float)
 
fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, cmap=plt.cm.jet, marker = '^', lw=0.2,label = 'Fixed injection',alpha=0.9)
im = ax.scatter(x1, y1, c=c1, cmap=plt.cm.jet, marker = '*', lw=0.2,label = 'Variable injection',alpha=0.9)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(340, 380)
cbar.ax.set_ylabel('Discharge temperature [K]')
#ax.text(0.68,0.05,'Markersize (injection superheat) ${:0.0g}$'.format(min(s))+' to {:0.0f} K'.format(max(s1)),ha='center',va='center',transform = ax.transAxes,fontsize = 7)
#plt.ylim(30,100)
#plt.xlim(0,24.875)
plt.ylabel('Condensation tempearture [K]')
plt.xlabel('Evaporation temperature [K]')
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()     
plt.savefig('T_dis_Tinj_Tcond_Tevap_FV.pdf')
#plt.show()
plt.close()


################################################################################
# m_ratio / T_inj_sh /T_cond / T_evap 
################################################################################
#assign axes
y = np.array(df[0:22]['T_cond [K]'], dtype=float)
x = np.array(df[0:22]['T_evap [K]'], dtype=float)
c = np.array(df[0:22]['Actual Injection Ratio'], dtype=float) * 100
s = np.array(df[0:22]['Injection Superheat (K)'], dtype=float)
y1 = np.array(df[22:]['T_cond [K]'], dtype=float)
x1 = np.array(df[22:]['T_evap [K]'], dtype=float)
c1 = np.array(df[22:]['Actual Injection Ratio'], dtype=float) * 100
s1 = np.array(df[22:]['Injection Superheat (K)'], dtype=float)
 
fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, cmap=plt.cm.jet, marker = '^', lw=0.2, label = 'Fixed injection',alpha=0.9)
im = ax.scatter(x1, y1, c=c1, cmap=plt.cm.jet, marker = '*', lw=0.2, label = 'Variable injection',alpha=0.9)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(0, 30)
cbar.ax.set_ylabel('$\dot m_{inj}/\dot m_{suc}$ [\%]')
#ax.text(0.68,0.05,'Markersize (injection superheat) ${:0.0g}$'.format(min(s))+' to {:0.0f} K'.format(max(s1)),ha='center',va='center',transform = ax.transAxes,fontsize = 7)
#plt.ylim(30,100)
#plt.xlim(0,24.875)
plt.ylabel('Condensation tempearture [K]')
plt.xlabel('Evaporation temperature [K]')
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()
plt.savefig('minj_Tinj_Tcond_Tevap_FV.pdf')
#plt.show()
plt.close()


################################################################################
# f_loss / T_inj_sh /T_cond / T_evap 
################################################################################
#assign axes
y = np.array(df[0:22]['T_cond [K]'], dtype=float)
x = np.array(df[0:22]['T_evap [K]'], dtype=float)
c = np.array(df[0:22]['actual heat loss'], dtype=float)
s = np.array(df[0:22]['Injection Superheat (K)'], dtype=float)
y1 = np.array(df[22:]['T_cond [K]'], dtype=float)
x1 = np.array(df[22:]['T_evap [K]'], dtype=float)
c1 = np.array(df[22:]['actual heat loss'], dtype=float)
s1 = np.array(df[22:]['Injection Superheat (K)'], dtype=float)
  
fig, ax = plt.subplots()
im = ax.scatter(x, y, c=c, cmap=plt.cm.jet, marker = '^', lw=0.2, label = 'Fixed injection',alpha=0.9)
im = ax.scatter(x1, y1, c=c1, cmap=plt.cm.jet, marker = '*', lw=0.2, label = 'Variable injection',alpha=0.9)
# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
# set the color limits
im.set_clim(2, 7)
cbar.ax.set_ylabel('$f_{loss}$ [\%]')
#ax.text(0.68,0.05,'Markersize (injection superheat) ${:0.0g}$'.format(min(s))+' to {:0.0f} K'.format(max(s1)),ha='center',va='center',transform = ax.transAxes,fontsize = 7)
#plt.ylim(30,100)
#plt.xlim(0,24.875)
plt.ylabel('Condensation tempearture [K]')
plt.xlabel('Evaporation temperature [K]')
leg=plt.legend(loc='upper right',scatterpoints=1)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout()
plt.savefig('floss_Tinj_Tcond_Tevap_FV.pdf')
#plt.show()
plt.close()
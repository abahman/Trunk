import os,sys
import numpy as np
import math
from math import log, pi
import pylab
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.mlab as ml
import pandas as pd
mpl.style.use('classic')
mpl.style.use('Elsevier.mplstyle')
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['figure.figsize'] = [6,4]
mpl.rcParams['legend.labelspacing'] = 0.2
mpl.rcParams['legend.numpoints'] = 1

# #===============================================================================
# # Latex render
# #===============================================================================
# #mpl.use('pgf')
#  
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
# #===============================================================================
# # END of Latex render
# #===============================================================================
    
def savefigs(name):
    #plt.savefig(name+'.eps')
    plt.savefig(name+'.pdf')
    #plt.savefig(name+'.png',dpi=600)
    #plt.show()

#########################
##### quenching_curve #######
#########################    
#import data from excel file
df1 = pd.read_excel('Results.xlsx',sheet_name='Figure 3',header=0) #file name
#assign axes
x1 = df1['Smooth_t'][0:2411]
y1 = df1['Smooth_Tc'][0:2411]
y11 = df1['Smooth_Ts'][0:2411]

x2 = df1['Threaded_t'][0:2411]
y2 = df1['Threaded_Tc'][0:2411]
y22 = df1['Threaded_Ts'][0:2411]

x3 = df1['Knurled_t'][0:2606]
y3 = df1['Knurled_Tc'][0:2606]
y33 = df1['Knurled_Ts'][0:2606]
  

plt.plot(x1,y1,'o',label=r'Smooth ($T_{c}$)',markeredgecolor='r',markerfacecolor='none',markersize=5,markevery=12)    
plt.plot(x1,y11,'r-',label=r'Smooth ($T_{s}$)',linewidth=2.5)    

plt.plot(x2,y2,'s',label=r'Threaded ($T_{c}$)',markeredgecolor='g',markerfacecolor='none',markersize=5,markevery=15)
plt.plot(x2,y22,'g--',label=r'Threaded ($T_{s}$)',linewidth=2.5)

plt.plot(x3,y3,'^',label=r'Knurled ($T_{c}$)',markeredgecolor='b',markerfacecolor='none',markersize=5,markevery=12)
plt.plot(x3,y33,'b.-',label=r'Knurled ($T_{s}$)',linewidth=2.5)
  
plt.ylim(100,600)
plt.xlim(0,70)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'Time [sec]')
plt.ylabel(r'Temperature [$\degree$C]') #{\textdegree}C
leg = plt.legend(loc='best',fancybox=False,numpoints=1, markerscale=1.25)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('quenching_curve')
# plt.show()
plt.close()
  
  
#########################
##### boiling_curve #######
#########################
#import data from excel file
df2 = pd.read_excel('Results.xlsx',sheet_name='Figure 4',header=0) #file name
#assign axes
x1 = df2['Smooth_delT'][0:2411]
y1 = df2['Smooth_q'][0:2411]/1000 #convert W to kW
x2 = df2['Threaded_delT'][0:2411]
y2 = df2['Threaded_q'][0:2411]/1000 #convert W to kW
x3 = df2['Knurled_delT'][0:2606]
y3 = df2['Knurled_q'][0:2606]/1000 #convert W to kW
  
# other statistics
y11 = df2['Smooth_q'][0:2411].rolling(15).mean()
y22 = df2['Threaded_q'][0:2411].rolling(15).mean()
y33 = df2['Knurled_q'][0:2606].rolling(15).mean()
  
plt.plot(x1, y1,'o',markerfacecolor='none',markeredgecolor='r',label=r'Smooth',markersize=5,markevery=2)#markeredgewidth=0.1,
plt.plot(x1, y11/1000,'r-',linewidth=2.5,label=r'Smooth (averaged)')
plt.plot(x2, y2,'s',markerfacecolor='none',markeredgecolor='g',label=r'Threaded',markersize=5,markevery=2)#markeredgewidth=0.1,
plt.plot(x2, y22/1000,'g--',linewidth=2.5,label=r'Threaded (averaged)')
plt.plot(x3, y3,'^',markerfacecolor='none',markeredgecolor='b',label=r'Knurled',markersize=5,markevery=2)#markeredgewidth=0.1,
plt.plot(x3, y33/1000,'b.-',linewidth=2.5,label=r'Knurled (averaged)')
  
plt.ylim(0,600)
plt.xlim(50,450)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$\Delta T_{w}$ [$\degree$C]')
plt.ylabel(r'Surface heat flux [kW/m$^2$]') #$q_{s}\H$
leg = plt.legend(loc='best',fancybox=False,numpoints=1, markerscale=1.25)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('boiling_curve')
# plt.show()
plt.close()
  
  
#########################
##### HTC #######
#########################
#import data from excel file
df3 = pd.read_excel('Results.xlsx',sheet_name='Figure 5',header=0) #file name
#assign axes
x1 = df3['Smooth_delT'][0:2411]
y1 = df3['Smooth_h'][0:2411]#/1000 #convert W to kW
x2 = df3['Threaded_delT'][0:2411]
y2 = df3['Threaded_h'][0:2411]#/1000 #convert W to kW
x3 = df3['Knurled_delT'][0:2606]
y3 = df3['Knurled_h'][0:2606]#/1000 #convert W to kW
  
# other statistics
y11 = df3['Smooth_h'][0:2411].rolling(15).mean()
y22 = df3['Threaded_h'][0:2411].rolling(15).mean()
y33 = df3['Knurled_h'][0:2606].rolling(15).mean()

#Calculate the polynomial for smooth
coefficients = np.polyfit(x1[0:1665], y1[0:1665], 2) #the end value is the T_min value so the fit will avoid including the otehr data
poly = np.poly1d(coefficients)
#Calculate new x and y values
new_x1 = x1[0:1665]#np.linspace(x1[0], x1[-1])
new_y1 = poly(new_x1)

#Calculate the polynomial for threaded
coefficients = np.polyfit(x2[0:1345], y2[0:1345], 2)
poly = np.poly1d(coefficients)
#Calculate new x and y values
new_x2 = x2[0:1345]
new_y2 = poly(new_x2)

#Calculate the polynomial for knurled
coefficients = np.polyfit(x3[0:783], y3[0:783], 2)
poly = np.poly1d(coefficients)
#Calculate new x and y values
new_x3 = x3[0:783]
new_y3 = poly(new_x3)

#Plot the polynomial fit
# plt.plot(x1, y1,'o',markerfacecolor='none',markeredgecolor='r',label=r'Smooth',markersize=5,markevery=10)
plt.plot(new_x1, new_y1,'r-',linewidth=2.5,label=r'Smooth')
# plt.plot(x1, y11/1000,'r-',linewidth=2.5,label=r'Smooth (averaged)')
# plt.plot(x2, y2,'s',markerfacecolor='none',markeredgecolor='g',label=r'Threaded',markersize=5,markevery=10)
plt.plot(new_x2, new_y2,'g--',linewidth=2.5,label=r'Threaded')
# plt.plot(x2, y22/1000,'g--',linewidth=2.5,label=r'Threaded (averaged)')
# plt.plot(x3, y3,'^',markerfacecolor='none',markeredgecolor='b',label=r'Knurled',markersize=5,markevery=10)
plt.plot(new_x3, new_y3,'b:',linewidth=2.5,label=r'Knurled')
# plt.plot(x3, y33/1000,'b.-',linewidth=2.5,label=r'Knurled (averaged)')
  
plt.ylim(0,600)
plt.xlim(200,450)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$\Delta T_{w}$ [$\degree$C]')
plt.ylabel(r'HTC [W/m$^2$-K]')
leg = plt.legend(loc='best',fancybox=False,numpoints=1, markerscale=1.25)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('HTC')
# plt.show()
plt.close()
  
  
#########################
##### film_thickness #######
#########################
#import data from excel file
df4 = pd.read_excel('Results.xlsx',sheet_name='Figure 6 new',header=0) #file name
#assign axes
x1 = df4['Smooth_delT'][0:1667]
y1 = df4['Smooth_del'][0:1667]
x2 = df4['Threaded_delT'][0:1345]
y2 = df4['Threaded_del'][0:1345]
x3 = df4['Knurled_delT'][0:783]
y3 = df4['Knurled_del'][0:783]
  
# other statistics
y11 = df4['Smooth_del'][0:1667].rolling(30).mean()
y22 = df4['Threaded_del'][0:1345].rolling(30).mean()
y33 = df4['Knurled_del'][0:783].rolling(30).mean()
  
plt.plot(x1, y1,'o',markerfacecolor='none',markeredgecolor='r',label=r'Smooth',markersize=5,markevery=3)#markeredgewidth=0.1,
# plt.plot(x1, y11,'r-',linewidth=2.5,label=r'Smooth (averaged)')
plt.axvline(x=217,color='r',ls='-',linewidth=2.5,)

plt.plot(x2, y2,'s',markerfacecolor='none',markeredgecolor='g',label=r'Threaded',markersize=5,markevery=3)#markeredgewidth=0.1,
# plt.plot(x2, y22,'g--',linewidth=2.5,label=r'Threaded (averaged)')
plt.axvline(x=230,color='g',ls='--',linewidth=2.5,)

plt.plot(x3, y3,'^',markerfacecolor='none',markeredgecolor='b',label=r'Knurled',markersize=5,markevery=3)#markeredgewidth=0.1,
# plt.plot(x3, y33,'b.-',linewidth=2.5,label=r'Knurled (averaged)')
plt.axvline(x=293,color='b',ls=':',linewidth=2.5,)


plt.annotate('Transition boiling',xy=(0,0),xytext=(65,25),annotation_clip=False)
plt.annotate('Film boiling',xy=(0,0),xytext=(325,25),annotation_clip=False)
# plt.annotate('', xy=(300,25), xytext=(350,25), arrowprops=dict(facecolor='blue',arrowstyle='<|-',shrinkB=2.75,lw=1.0))


plt.ylim(0,300)
plt.xlim(50,450)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$\Delta T_{w}$ [$\degree$C]')
plt.ylabel(r'Film thickness [$\mu$m]')
leg = plt.legend(loc='best',fancybox=False,numpoints=1, markerscale=1.25)
frame  = leg.get_frame()
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('film_thickness')
plt.show()
plt.close()
  
  
#########################
##### quenching_time #######
#########################
#import data from excel file
df5 = pd.read_excel('Results.xlsx',sheet_name='Figure 8',header=0) #file name
#assign axes
x = df5['EXP'][0:10]
y1 = df5['Smooth'][0:10]
y2 = df5['Threaded'][0:10]
y3 = df5['Knurled'][0:10]
  
plt.plot(x, y1,'ro',label=r'Smooth',markerfacecolor='white',markeredgecolor='r') 
plt.plot(x, y2,'gs',label=r'Threaded',markerfacecolor='white',markeredgecolor='g')
plt.plot(x, y3,'b^',label=r'Knurled',markerfacecolor='white',markeredgecolor='b')
  
# plt.ylim(10,50)
plt.xlim(0,11)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
           [r'', r'1', r'2', r'3',r'4', r'5', r'6', r'7', r'8', r'9', r'10', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'Number of quenching experiment')
plt.ylabel(r'Quenching time [sec]')
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('quenching_time')
# plt.show()
plt.close()
  
  
#########################
##### T_min #######
#########################
#import data from excel file
df6 = pd.read_excel('Results.xlsx',sheet_name='Figure 9',header=0) #file name
#assign axes
x = df6['EXP'][0:10]
y1 = df6['Smooth'][0:10]
y2 = df6['Threaded'][0:10]
y3 = df6['Knurled'][0:10]
  
plt.plot(x, y1,'ro',label=r'Smooth',markerfacecolor='white',markeredgecolor='r') 
plt.plot(x, y2,'gs',label=r'Threaded',markerfacecolor='white',markeredgecolor='g')
plt.plot(x, y3,'b^',label=r'Knurled',markerfacecolor='white',markeredgecolor='b')
  
# plt.ylim(0,60)
plt.xlim(0,11)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
           [r'', r'1', r'2', r'3',r'4', r'5', r'6', r'7', r'8', r'9', r'10', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'Number of quenching experiment')
plt.ylabel(r'$T_{min}$ [$\degree$C]')
leg = plt.legend(loc='best',fancybox=False,numpoints=1)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('T_min')
# plt.show()
plt.close()


#########################
##### comparison_barplot #######
#########################
#import data from excel file
fig=plt.figure(figsize=(6,6))

df7 = pd.read_excel('Results.xlsx',sheet_name='Figure 13',header=0) #file name
#assign axes
x = df7['study'][0:15]
y = df7['Tmin_cal'][0:15]

x_pos = np.arange(len(x))
barlist=plt.bar(x_pos,y,color='black',width=0.9,linewidth=0.9,align='center',alpha=0.9)
barlist[0].set_color('r')
barlist[1].set_color('g')
barlist[2].set_color('b')

# plt.ylim(0,9000)
# plt.xlim(-2,16)
plt.xticks(x_pos,x,rotation=90, fontsize=12)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
# plt.xlabel(r'Test condition')
plt.ylabel(r'$T_{min}$ [$\degree$C]')
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('comparison_barplot')
# plt.show()
plt.close()

#########################
##### biot_number #######
#########################
#import data from excel file
df8 = pd.read_excel('Results.xlsx',sheet_name='Appendix',header=0) #file name
#assign axes
x1 = df8['Smooth_delT'][0:2411]
y1 = df8['Smooth_Bi'][0:2411]
x2 = df8['Threaded_delT'][0:2411]
y2 = df8['Threaded_Bi'][0:2411]
x3 = df8['Knurled_delT'][0:2606]
y3 = df8['Knurled_Bi'][0:2606]
 
# other statistics
y11 = df8['Smooth_Bi'][0:2411].rolling(15).mean()
y22 = df8['Threaded_Bi'][0:2411].rolling(15).mean()
y33 = df8['Knurled_Bi'][0:2606].rolling(15).mean()
 
plt.plot(x1, y1,'o',markerfacecolor='none',markeredgecolor='r',label=r'Smooth',markersize=5,markevery=2)#markeredgewidth=0.1,
plt.plot(x1, y11,'r-',linewidth=2.5,label=r'Smooth (averaged)')
 
plt.plot(x2, y2,'s',markerfacecolor='none',markeredgecolor='g',label=r'Threaded',markersize=5,markevery=2)#markeredgewidth=0.1,
plt.plot(x2, y22,'g--',linewidth=2.5,label=r'Threaded (averaged)')
 
plt.plot(x3, y3,'^',markerfacecolor='none',markeredgecolor='b',label=r'Knurled',markersize=5,markevery=2)#markeredgewidth=0.1,
plt.plot(x3, y33,'b.-',linewidth=2.5,label=r'Knurled (averaged)')
 
plt.ylim(0,0.8)
plt.xlim(50,450)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#            [r'', r'1', r'2', r'3',r'4/A', r'5', r'6', r'B', r'C', r''])
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='on',      # ticks along the bottom edge are on
#     top='off',         # ticks along the top edge are off
#     labelbottom='on') # labels along the bottom edge are off
plt.xlabel(r'$\Delta T_{w}$ [$\degree$C]')
plt.ylabel(r'Biot number [$-$]')
leg = plt.legend(loc='best',fancybox=False,numpoints=1, markerscale=1.25)
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('biot_number')
# plt.show()
plt.close()
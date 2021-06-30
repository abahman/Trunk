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
    plt.savefig('images/'+name+'.pdf')
    plt.savefig(name+'.png',dpi=600)
    #plt.show()


# #########################
# ##### optimized results (COP)  #######
# #########################  
# Set data
df = pd.DataFrame({
'group': ['A','B','AB'],
'R-32': [2.744,2.425,2.733],
'R-290': [5.188,4.393,5.16],
'R-410A': [4.073,2.449,3.999],
'R-454A': [4.972,4.649,4.911],
'R-452B': [3.963,2.497,3.913]
})
 
# ------- PART 1: Create background
 
# number of variable
categories=list(df)[1:]
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialize the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([1,2,3,4,5,6], ["1","2","3","4","5"," "], color="grey", size=12)
plt.ylim(0,6)
 

# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't make a loop, because plotting more than 3 groups makes the chart unreadable
 
# Ind1
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, color='yellow', linewidth=1.5, linestyle='solid', label="Point A")
ax.fill(angles, values, color='yellow', alpha=0.1)
 
# Ind2
values=df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, color='orange', linewidth=1.5, linestyle='solid', label="Point B")
ax.fill(angles, values, color='orange', alpha=0.1)

# Ind3
values=df.loc[2].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, color='brown', linewidth=1.5, linestyle='solid', label="Point AB")
ax.fill(angles, values, color='brown', alpha=0.1)


# Go through labels and adjust alignment based on where
# it is in the circle.
for label, angle in zip(ax.get_xticklabels(), angles):
  if angle in (0, np.pi):
    label.set_horizontalalignment('center')
  elif 0 < angle < np.pi:
    label.set_horizontalalignment('left')
  else:
    label.set_horizontalalignment('right')
    
# Add legend
leg = plt.legend(loc='upper right',fancybox=False,numpoints=1,bbox_to_anchor=(1.35, 1.05))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('optimized_results_COP_radar')
plt.show()
plt.close()





# #########################
# #####  optimized results (UCH) #######
# #########################
# Set data
df = pd.DataFrame({
'group': ['A','B','AB'],
'R-32': [0.193,0.188,0.191],
'R-290': [0.234,0.229,0.232],
'R-410A': [0.231,0.221,0.226],
'R-454A': [0.231,0.227,0.228],
'R-452B': [0.221,0.208,0.214]
})
 
# ------- PART 1: Create background
 
# number of variable
categories=list(df)[1:]
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialize the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.05,0.1,0.15,0.2,0.25,0.3], ["0.05","0.1","0.15","0.2","0.25"," "], color="grey", size=12)
plt.ylim(0,0.3)
 

# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't make a loop, because plotting more than 3 groups makes the chart unreadable
 
# Ind1
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, color='m', linewidth=1.5, linestyle='solid', label="Point A")
ax.fill(angles, values, color='m', alpha=0.1)
 
# Ind2
values=df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, color='c', linewidth=1.5, linestyle='solid', label="Point B")
ax.fill(angles, values, color='c', alpha=0.1)

# Ind3
values=df.loc[2].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, color='r', linewidth=1.5, linestyle='solid', label="Point AB")
ax.fill(angles, values, color='r', alpha=0.1)


# Go through labels and adjust alignment based on where
# it is in the circle.
for label, angle in zip(ax.get_xticklabels(), angles):
  if angle in (0, np.pi):
    label.set_horizontalalignment('center')
  elif 0 < angle < np.pi:
    label.set_horizontalalignment('left')
  else:
    label.set_horizontalalignment('right')
    
# Add legend
leg = plt.legend(loc='upper right',fancybox=False,numpoints=1,bbox_to_anchor=(1.35, 1.05))
frame  = leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('optimized_results_UCH_radar')
plt.show()
plt.close()
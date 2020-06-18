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
mpl.rcParams['figure.figsize'] = [4,4]
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
    #plt.savefig(name+'.png',dpi=600)
    #plt.show()


# #########################
# ##### Figure 3a #######
# #########################
# def arrowed_spines(fig, ax):
# 
#     xmin, xmax = ax.get_xlim() 
#     ymin, ymax = ax.get_ylim()
# 
#     # removing the default axis on all sides:
#     for side in ['bottom','right','top','left']:
#         ax.spines[side].set_visible(False)
# 
#     # removing the axis ticks
#     plt.xticks([]) # labels 
#     plt.yticks([])
#     ax.xaxis.set_ticks_position('none') # tick markers
#     ax.yaxis.set_ticks_position('none')
# 
#     # get width and height of axes object to compute 
#     # matching arrowhead length and width
#     dps = fig.dpi_scale_trans.inverted()
#     bbox = ax.get_window_extent().transformed(dps)
#     width, height = bbox.width, bbox.height
# 
#     # manual arrowhead width and length
#     hw = 1./30.*(ymax-ymin) 
#     hl = 1./30.*(xmax-xmin)
#     lw = 1.5 # axis line width
#     ohg = 0.0 # arrow overhang
# 
#     # compute matching arrowhead length and width
#     yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
#     yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height
# 
#     # draw x and y axis
#     ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
#              head_width=hw, head_length=hl, overhang = ohg, 
#              length_includes_head= True, clip_on = False) 
# 
#     ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
#              head_width=yhw, head_length=yhl, overhang = ohg, 
#              length_includes_head= True, clip_on = False)
# 
# 
# # plot
# fig = plt.gcf()
# ax = plt.gca()
# 
# from matplotlib.path import Path
# import matplotlib.patches as patches
# pathdata = [
#     (Path.MOVETO, (0.9, 0.1)),
#     (Path.CURVE4, (0.015, 0.05)),
#     (Path.CURVE4, (0.025, 0.9)),
#     (Path.CURVE4, (0.275, 0.9)),
#     (Path.CURVE4, (0.8, 0.8)),
#     (Path.CURVE4, (0.9, 0.7)),
#     (Path.CURVE4, (0.95, 0.1)),
#     (Path.CLOSEPOLY, (0.5, 0.1)),
#     ]
# codes, verts = zip(*pathdata)
# path = Path(verts, codes)
# patch = patches.PathPatch(path, facecolor='none', lw=1)
# ax.add_patch(patch)
# 
# x_loc = np.array([0.20469,
# 0.22969,
# 0.28281,
# 0.32109,
# 0.43437,
# 0.37578,
# 0.52969,
# 0.55703,
# 0.47266,
# 0.38203,
# 0.51016,
# 0.40547,
# 0.54766,
# 0.63750,
# 0.60781,
# 0.63516,
# 0.73047,
# 0.82578,
# 0.78437,
# 0.69844,
# 0.77578,
# 0.85000,
# 0.73203,
# 0.29609]) 
# y_loc = np.array([0.75233,
# 0.62517,
# 0.52122,
# 0.76865,
# 0.68444,
# 0.59531,
# 0.77099,
# 0.66519,
# 0.52173,
# 0.41027,
# 0.37948,
# 0.26354,
# 0.25591,
# 0.33622,
# 0.46515,
# 0.59600,
# 0.51174,
# 0.52089,
# 0.38110,
# 0.26521,
# 0.16755,
# 0.26651,
# 0.62562,
# 0.42606])
# 
# for i in range(len(x_loc)):
#     ax.add_patch(plt.Circle((x_loc[i], y_loc[i]), 0.04, color='r', alpha=0.9))
# 
# #add arrowheads to a and y axes
# arrowed_spines(fig, ax)
# 
# ax.annotate('Initialization\n of population', xy=(0.28, 0.28), xytext=(0.05, 0.05),
#             arrowprops=dict(facecolor='black',lw=0.0),
#             )
# 
# ax.set_xlabel(r'$f_{2} (x)$',position=(0.95, 0),horizontalalignment='right')
# ax.set_ylabel(r'$f_{1} (x)$',position=(0, 0.95),horizontalalignment='right')
# # leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# # frame  = leg.get_frame()  
# # frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig3a')
# plt.show()
# plt.close()













# #########################
# ##### Figure 3b #######
# #########################
# def arrowed_spines(fig, ax):
#  
#     xmin, xmax = ax.get_xlim() 
#     ymin, ymax = ax.get_ylim()
#  
#     # removing the default axis on all sides:
#     for side in ['bottom','right','top','left']:
#         ax.spines[side].set_visible(False)
#  
#     # removing the axis ticks
#     plt.xticks([]) # labels 
#     plt.yticks([])
#     ax.xaxis.set_ticks_position('none') # tick markers
#     ax.yaxis.set_ticks_position('none')
#  
#     # get width and height of axes object to compute 
#     # matching arrowhead length and width
#     dps = fig.dpi_scale_trans.inverted()
#     bbox = ax.get_window_extent().transformed(dps)
#     width, height = bbox.width, bbox.height
#  
#     # manual arrowhead width and length
#     hw = 1./30.*(ymax-ymin) 
#     hl = 1./30.*(xmax-xmin)
#     lw = 1.5 # axis line width
#     ohg = 0.0 # arrow overhang
#  
#     # compute matching arrowhead length and width
#     yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
#     yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height
#  
#     # draw x and y axis
#     ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
#              head_width=hw, head_length=hl, overhang = ohg, 
#              length_includes_head= True, clip_on = False) 
#  
#     ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
#              head_width=yhw, head_length=yhl, overhang = ohg, 
#              length_includes_head= True, clip_on = False)
#  
#  
# # plot
# fig = plt.gcf()
# ax = plt.gca()
#  
# from matplotlib.path import Path
# import matplotlib.patches as patches
# pathdata = [
#     (Path.MOVETO, (0.9, 0.1)),
#     (Path.CURVE4, (0.015, 0.05)),
#     (Path.CURVE4, (0.025, 0.9)),
#     (Path.CURVE4, (0.275, 0.9)),
#     (Path.CURVE4, (0.8, 0.8)),
#     (Path.CURVE4, (0.9, 0.7)),
#     (Path.CURVE4, (0.95, 0.1)),
#     (Path.CLOSEPOLY, (0.5, 0.1)),
#     ]
# codes, verts = zip(*pathdata)
# path = Path(verts, codes)
# patch = patches.PathPatch(path, facecolor='none', lw=1)
# ax.add_patch(patch)
#  
# x_loc = np.array([0.20746,
# 0.46029,
# 0.64830,
# 0.35737,
# 0.54943,
# 0.72771,
# 0.61021,
# 0.82172,
# 0.31524,
# 0.24311,
# 0.50081,
# 0.41167,
# 0.87196]) 
# y_loc = np.array([0.54108,
# 0.28110,
# 0.15581,
# 0.66483,
# 0.48843,
# 0.34454,
# 0.61860,
# 0.47660,
# 0.40829,
# 0.80878,
# 0.70493,
# 0.82845,
# 0.26568])
#  
# #draw colored circles
# for i in range(3):
#     ax.add_patch(plt.Circle((x_loc[i], y_loc[i]), 0.04, color='b', alpha=0.9))
#     ax.annotate("1", xy=(x_loc[i], y_loc[i]), fontsize=10, ha="center",va='center')
# for i in range(3):
#     ax.add_patch(plt.Circle((x_loc[3+i], y_loc[3+i]), 0.04, color='g', alpha=0.9))
#     ax.annotate("2", xy=(x_loc[3+i], y_loc[3+i]), fontsize=10, ha="center",va='center')
# for i in range(2):
#     ax.add_patch(plt.Circle((x_loc[6+i], y_loc[6+i]), 0.04, color='r', alpha=0.9))
#     ax.annotate("3", xy=(x_loc[6+i], y_loc[6+i]), fontsize=10, ha="center",va='center')
# ax.add_patch(plt.Circle((x_loc[8], y_loc[8]), 0.04, color='b', alpha=0.9))
# ax.annotate("1", xy=(x_loc[8], y_loc[8]), fontsize=10, ha="center",va='center')
# ax.add_patch(plt.Circle((x_loc[9], y_loc[9]), 0.04, color='g', alpha=0.9))
# ax.annotate("2", xy=(x_loc[9], y_loc[9]), fontsize=10, ha="center",va='center')
# for i in range(2):
#     ax.add_patch(plt.Circle((x_loc[10+i], y_loc[10+i]), 0.04, color='r', alpha=0.9))
#     ax.annotate("3", xy=(x_loc[10+i], y_loc[10+i]), fontsize=10, ha="center",va='center')
# ax.add_patch(plt.Circle((x_loc[12], y_loc[12]), 0.04, color='g', alpha=0.9))
# ax.annotate("2", xy=(x_loc[12], y_loc[12]), fontsize=10, ha="center",va='center')
#  
#    
# #add arrowheads to a and y axes
# arrowed_spines(fig, ax)
#  
# ax.annotate('', xy=(0.4, -0.05),xytext=(0.8,-0.05),                     #draws an arrow from one set of coordinates to the other
#             arrowprops=dict(facecolor='k',lw=0.0),   #sets style of arrow and colour
#             annotation_clip=False)                               #This enables the arrow to be outside of the plot
#  
# ax.annotate('Minimization',xy=(0, -0.05),xytext=(0,-0.065),               #Adds another annotation for the text that you want
#             annotation_clip=False)
#  
# ax.annotate('', xy=(-0.05, 0.4),xytext=(-0.05,0.8),                     #draws an arrow from one set of coordinates to the other
#             arrowprops=dict(facecolor='k',lw=0.0),   #sets style of arrow and colour
#             annotation_clip=False)                               #This enables the arrow to be outside of the plot
#  
# ax.annotate('Minimization',xy=(-0.075, 0),xytext=(-0.075,0.35), rotation=90,              #Adds another annotation for the text that you want
#             annotation_clip=False)
#  
#  
# ax.set_xlabel(r'$f_{2} (x)$',position=(0.95, 0),horizontalalignment='right')
# ax.set_ylabel(r'$f_{1} (x)$',position=(0, 0.95),horizontalalignment='right')
# # leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# # frame  = leg.get_frame()  
# # frame.set_linewidth(0.5)
# plt.tight_layout(pad=0.2) 
# savefigs('fig3b')
# plt.show()
# plt.close()






#########################
##### Figure 3c #######
#########################
def arrowed_spines(fig, ax):

    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)

    # removing the axis ticks
    plt.xticks([]) # labels 
    plt.yticks([])
    ax.xaxis.set_ticks_position('none') # tick markers
    ax.yaxis.set_ticks_position('none')

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./30.*(ymax-ymin) 
    hl = 1./30.*(xmax-xmin)
    lw = 1.5 # axis line width
    ohg = 0.0 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
             head_width=hw, head_length=hl, overhang = ohg, 
             length_includes_head= True, clip_on = False) 

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
             head_width=yhw, head_length=yhl, overhang = ohg, 
             length_includes_head= True, clip_on = False)


# plot
fig = plt.gcf()
ax = plt.gca()


# from matplotlib.path import Path
# import matplotlib.patches as patches
# pathdata = [
#     (Path.MOVETO, (0.9, 0.1)),
#     (Path.CURVE4, (0.015, 0.05)),
#     (Path.CURVE4, (0.025, 0.9)),
#     (Path.CURVE4, (0.275, 0.9)),
#     (Path.CURVE4, (0.8, 0.8)),
#     (Path.CURVE4, (0.9, 0.7)),
#     (Path.CURVE4, (0.95, 0.1)),
#     (Path.CLOSEPOLY, (0.5, 0.1)),
#     ]
# codes, verts = zip(*pathdata)
# path = Path(verts, codes)
# patch = patches.PathPatch(path, facecolor='none', lw=1)
# ax.add_patch(patch)

x = np.array([0.000,
0.04575,
0.14592,
0.28366,
0.35944,
0.44353,
0.60421,
0.69534,
0.86364,
0.96938]) 
y = np.array([0.85603,
0.74573,
0.69064,
0.62524,
0.51152,
0.35124,
0.24707,
0.20922,
0.04816,
0.00000])

x2 = np.array([0.13697,
0.26011,
0.56637,
0.71152,
0.80263,
0.94796]) 
y2 = np.array([0.82339,
0.77609,
0.88066,
0.45492,
0.37914,
0.21978])

#add arrowheads to a and y axes
arrowed_spines(fig, ax)

#draw dahsed box
x3=np.array([x[2],x[4],x[4],x[2],x[2]])
y3=np.array([y[2],y[2],y[4],y[4],y[2]])
plt.plot(x3,y3,'--k')
#plot blue dots
x_new = np.linspace(x.min(), x.max(),500)
from scipy.interpolate import interp1d
f = interp1d(x, y, kind='quadratic')
y_smooth=f(x_new)
ax.plot(x_new,y_smooth,'b')
plt.plot(x[1:9],y[1:9],'bo',markersize=10,markeredgecolor='b')
#plot red dots
plt.plot(x2,y2,'ro',markersize=10,markeredgecolor='r',markerfacecolor='r')
#annotation
ax.annotate('$i$',xy=(x[3], y[3]-0.05),xytext=(x[3]-0.05, y[3]-0.05),annotation_clip=False)
ax.annotate('$i-1$',xy=(x[2], y[2]-0.05),xytext=(x[2]-0.05, y[2]+0.03),annotation_clip=False)
ax.annotate('$i+1$',xy=(x[4], y[4]-0.05),xytext=(x[4]+0.05, y[4]-0.025),annotation_clip=False)

#axes limits
ax.set_xlim((0,1))
ax.set_ylim((0,1))

ax.set_xlabel(r'$f_{2} (x)$',position=(0.95, 0),horizontalalignment='right')
ax.set_ylabel(r'$f_{1} (x)$',position=(0, 0.95),horizontalalignment='right')
# leg = plt.legend(loc='best',fancybox=False,numpoints=1)
# frame  = leg.get_frame()  
# frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2) 
savefigs('fig3c')
plt.show()
plt.close()

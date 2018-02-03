import pylab, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
mpl.style.use('classic')
mpl.style.use('Elsevier.mplstyle')
mpl.rcParams['mathtext.fontset'] = 'custom'

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

fig = plt.figure()
ax = fig.add_subplot(111)

#draw upper/lower lines
plt.plot(np.r_[0.0,5],np.r_[1,1],'k')
plt.plot(np.r_[0.0,5],np.r_[0.0,0.0],'k')
plt.plot(np.r_[0.0,5],np.r_[-1,-1],'k')

#gradient bottom
gradient2 = np.linspace(0, 1, 2)
gradient2 = np.vstack((gradient2, gradient2))
imgplot = pylab.imshow(gradient2, extent=[0,5,-1,0], alpha=0.9,aspect=4, cmap=plt.get_cmap('brg'))
imgplot.set_clim(0.1,2.0)

#gradient top
imgplot = pylab.imshow(gradient2, extent=[5,0,1,0], alpha=0.9,aspect=4, cmap=plt.get_cmap('brg'))
imgplot.set_clim(0.1,2.0)

#seperation line
plt.plot(np.r_[1,1],np.r_[-1,0],'k--')
plt.plot(np.r_[1,1],np.r_[0,1],'k-')

plt.plot(np.r_[2,2],np.r_[-1,0],'k-')
plt.plot(np.r_[2,2],np.r_[0,1],'k--')

plt.plot(np.r_[3,3],np.r_[-1,0],'k--')
plt.plot(np.r_[3,3],np.r_[0,1],'k-')

plt.plot(np.r_[4,4],np.r_[-1,0],'k-')
plt.plot(np.r_[4,4],np.r_[0,1],'k--')

#plot arrows
bbox_props = dict(boxstyle="rarrow", fc="w", ec="k", lw=1)
t1 = ax.text(-0.29, 0.5, "Cold stream", color="k", ha="center", va="center", rotation=0,
            size=10,
            bbox=bbox_props)
bb = t1.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.5)

t1 = ax.text(5.26, -0.5, "Hot stream", color="k", ha="center", va="center", rotation=0,
            size=10,
            bbox=bbox_props)
bb = t1.get_bbox_patch()
bb.set_boxstyle("larrow", pad=0.5)

#text
ax.text(0.56,0.5,'Superheated',ha='center',va='center',color="k",size=10)
ax.text(1.5,0.5,'Two-phase',ha='center',va='center',color="k",size=10)
ax.text(2.5,0.5,'Two-phase',ha='center',va='center',color="k",size=10)
ax.text(3.5,0.5,'Subcooled',ha='center',va='center',color="k",size=10)
ax.text(4.5,0.5,'Subcooled',ha='center',va='center',color="k",size=10)

ax.text(0.5,-0.5,'Subcooled',ha='center',va='center',color="k",size=10)
ax.text(1.5,-0.5,'Subcooled',ha='center',va='center',color="k",size=10)
ax.text(2.5,-0.5,'Two-phase',ha='center',va='center',color="k",size=10)
ax.text(3.5,-0.5,'Two-phase',ha='center',va='center',color="k",size=10)
ax.text(4.425,-0.5,'Superheated',ha='center',va='center',color="k",size=10)

plt.gca().set_xlim(-0.1,5.1)
plt.gca().axis('equal')
plt.gca().axis('off')
plt.savefig('economizer_section.pdf')
plt.show()
import pylab, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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

#fig = pylab.figure(figsize=(6,4))
fig = pylab.figure()
ax = fig.add_subplot(111)

y0=1.7 #shift of second axis
pylab.plot(np.r_[-0.01,4.01,4.01,-0.01,-0.01],np.r_[-y0,-y0,-y0+1,-y0+1,-y0],alpha=0)
pylab.plot(np.r_[0,4,4,0,0],np.r_[-y0,-y0,-y0+1,-y0+1,-y0],'k')
pylab.plot(np.r_[2.75,2.75],np.r_[-y0,-y0+1],'w-.')
pylab.plot(np.r_[1.25,1.25],np.r_[-y0,-y0+1],'w-.')
pylab.gca().text(0.625,-y0+0.5,'Subcooled',ha='center',va='center',color='w')
pylab.gca().text(2.0,-y0+0.5,'Two-Phase',ha='center',va='center',color='w')
pylab.gca().text(3.375,-y0+0.5,'Superheated',ha='center',va='center',color='w')
pylab.gca().text(2.0,0.2,'Ambient Air',ha='center',va='center')
pylab.gca().text(2.0,-2.6,'Discharged Air',ha='center',va='center')
pylab.gca().text(-1.25,-1.7,'Refrigerant\nto liquid line',ha='left',va='center')
pylab.gca().text(4.5,-.7,'Refrigerant\nfrom discharge\nline',ha='left',va='center')
# pylab.gca().text(2,-y0+1.1,'Superheated and Two-Phase Sections',ha='center',va='bottom')
# pylab.gca().text(3.5,-y0-0.1,'$w_{superheat}$',ha='center',va='center')
# pylab.gca().text(1.5,-y0-0.1,'$w_{two-phase}$',ha='center',va='center')

#plot arrows
bbox_props = dict(boxstyle="rarrow", fc="w", ec="k", lw=1)
t1 = ax.text(2, -0.25, " BCDE", color="w", ha="center", va="center", rotation=-90,
            size=6,
            bbox=bbox_props)
t2 = ax.text(2, -2.1, " BCDE", color="w", ha="center", va="center", rotation=-90,
            size=6,
            bbox=bbox_props)
t3 = ax.text(3.375, -0.25, " BCDE", color="w", ha="center", va="center", rotation=-90,
            size=6,
            bbox=bbox_props)
t4 = ax.text(3.375, -2.1, " BCDE", color="w", ha="center", va="center", rotation=-90,
            size=6,
            bbox=bbox_props)
t5 = ax.text(-0.5, -1.7+0.5, " BCDE", color="w", ha="center", va="center", rotation=180,
            size=6,
            bbox=bbox_props)
t6 = ax.text(4.5, -1.7+0.5, " BCDE", color="w", ha="center", va="center", rotation=180,
            size=6,
            bbox=bbox_props)
t7 = ax.text(0.625, -0.25, " BCDE", color="w", ha="center", va="center", rotation=-90,
            size=6,
            bbox=bbox_props)
t8 = ax.text(0.625, -2.1, " BCDE", color="w", ha="center", va="center", rotation=-90,
            size=6,
            bbox=bbox_props)
  
bb = t1.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.5)
bb = t2.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.5)
bb = t3.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.5)
bb = t4.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.5)
bb = t5.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.5)
bb = t6.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.5)
bb = t7.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.5)
bb = t8.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.5)

#plot color gradient 2nd plot
gradient2 = np.linspace(0, 1, 2)
gradient2 = np.vstack((gradient2, gradient2))
imgplot = pylab.imshow(gradient2, extent=[0,4.0,-1.7,-0.71], alpha=0.9,aspect=4, cmap=plt.get_cmap('brg'))
imgplot.set_clim(0.1,2.0)

pylab.gca().axis('equal')
pylab.gca().axis('off')
pylab.savefig('condenser_sections.pdf')
pylab.show()
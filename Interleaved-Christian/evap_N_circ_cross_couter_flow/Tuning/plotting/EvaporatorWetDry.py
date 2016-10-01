import pylab,numpy as np
from matplotlib.patches import FancyArrowPatch

#===============================================================================
# Latex render
#===============================================================================
import matplotlib as mpl
from numpy import integer
from numba.targets.randomimpl import f_impl
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

fig=pylab.figure(figsize=(8,4))

pylab.fill(np.r_[0,5,5,0,0],np.r_[-1,-1,0,0,-1],'b',alpha=0.1)
pylab.plot(np.r_[0,5],np.r_[1,1],'k')
pylab.plot(np.r_[0,5],np.r_[0,0],'k')
pylab.plot(np.r_[0,5],np.r_[-1,-1],'k')

pylab.plot(np.r_[3,3],np.r_[-1,1],'k--')

pylab.plot(3,0.5,'ko')
pylab.plot(3,-0.5,'ko')
pylab.text(3.05,0.5,'$T_{a,x}$',ha='left',va='center')
pylab.text(3.05,-0.5,'$T_{g,x}$',ha='left',va='center')

pylab.plot(0,0.5,'ko')
pylab.plot(0,-0.5,'ko')
pylab.text(0.05,0.5,'$T_{a,i}$',ha='left',va='center')
pylab.text(0.05,-0.5,'$T_{g,o}$',ha='left',va='center')

pylab.plot(5,0.5,'ko')
pylab.plot(5,-0.5,'ko')
pylab.text(5.05,0.5,'$T_{a,o}$',ha='left',va='center')
pylab.text(5.05,-0.5,'$T_{g,i}$',ha='left',va='center')

pylab.plot(5,0.5,'ko')
pylab.plot(5,-0.5,'ko')
pylab.text(5.05,0.5,'$T_{a,o}$',ha='left',va='center')
pylab.text(5.05,-0.5,'$T_{g,i}$',ha='left',va='center')
pylab.gca().add_patch(FancyArrowPatch((1,0.5),(0.5,0.5),arrowstyle='<|-',fc='k',ec='k',mutation_scale=20,lw=0.8))

#Wet line
pylab.plot(np.r_[3,5],np.r_[0.01,0.01],'b',lw=4)

pylab.plot(0,0,'ro')
pylab.text(-0.05,0,'$T_{i,s}$',ha='right',va='center')

pylab.plot(5,0,'ro')
pylab.text(5.05,0,'$T_{o,s}$',ha='left',va='center')

pylab.text(4,-1.1,'Wet Part',ha='center',va='top')
pylab.text(1.5,-1.1,'Dry Part',ha='center',va='top')
pylab.gca().add_patch(FancyArrowPatch((4,-0.5),(4.5,-0.5),arrowstyle='<|-',fc='k',ec='k',mutation_scale=20,lw=0.8))

pylab.gca().set_xlim(-0.1,5.1)
pylab.gca().axis('equal')
pylab.gca().axis('off')
pylab.savefig('evaporator_wet_dry.pdf')
pylab.show()
from __future__ import division
import matplotlib,pylab, numpy as np
from matplotlib.patches import FancyArrowPatch
import matplotlib as mpl
mpl.style.use('classic')

#===============================================================================
# Latex render
#===============================================================================
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

# fig=pylab.figure(figsize=(6,2))
# ax=fig.add_axes((0,0,1,1))

t=np.linspace(0,4*np.pi,100)
pylab.plot(t,np.cos(t))
    

pylab.gca().add_patch(FancyArrowPatch((4*np.pi,-1),(4*np.pi,1),arrowstyle='<|-|>',fc='k',ec='k',mutation_scale=10,lw=0.5))
pylab.text(4.05*np.pi,0,r' $2\hat a$',ha='left',va='center')
pylab.gca().add_patch(FancyArrowPatch((2*np.pi,1.1),(4*np.pi,1.1),arrowstyle='<|-|>',fc='k',ec='k',mutation_scale=10,lw=0.5))
pylab.text(3*np.pi,1.15,r' $\Lambda$',ha='center',va='bottom')

pylab.gca().axis('equal')
pylab.gca().axis('off')
pylab.savefig('PHEWavyPlate.pdf')
pylab.show()
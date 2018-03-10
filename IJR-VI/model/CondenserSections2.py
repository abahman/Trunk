import pylab, numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

fig=pylab.figure()
pylab.plot(np.r_[0.0,4.0,4.0,0.0,0.0],np.r_[0,0,1,1,0],'k')
pylab.plot(np.r_[0,4],np.r_[0.993,0.993],'k')

pylab.plot(np.r_[1,1],np.r_[0,1],'k-.')
pylab.plot(np.r_[3,3],np.r_[0,1],'k-.')

pylab.gca().text(2,0.5,'Two-Phase',ha='center',va='center')
pylab.gca().text(3.5,0.5,'Superheated',ha='center',va='center')
pylab.gca().text(0.5,0.5,'Subcooled',ha='center',va='center')
pylab.gca().text(2,1.1,'Superheated, Subcooled and Two-Phase Sections',ha='center',va='bottom')

pylab.gca().text(0.5,-0.1,'$w_{subcool}$',ha='center',va='center')
pylab.gca().text(2,-0.1,'$w_{two-phase}$',ha='center',va='center')
pylab.gca().text(3.5,-0.1,'$w_{superheat}$',ha='center',va='center')

#plot color gradient 1st plot
gradient = np.linspace(0, 1, 2)
gradient = np.vstack((gradient, gradient))
img = pylab.imshow(gradient, extent=[0,4.0,0,1.0], alpha=0.9, aspect=4, cmap=plt.get_cmap('brg'))
img.set_clim(0.1,2.0)

y0=1.7 #shift of second axis
pylab.plot(np.r_[0.0,4.0,4.0,0.0,0.0],np.r_[-y0,-y0,-y0+1,-y0+1,-y0],'k')
pylab.plot(np.r_[3,3],np.r_[-y0,-y0+1],'k-.')
pylab.gca().text(1.5,-y0+0.5,'Two-Phase',ha='center',va='center')
pylab.gca().text(3.5,-y0+0.5,'Superheated',ha='center',va='center')
pylab.gca().text(2,-y0+1.1,'Superheated and Two-Phase Sections',ha='center',va='bottom')

pylab.gca().text(3.5,-y0-0.1,'$w_{superheat}$',ha='center',va='center')
pylab.gca().text(1.5,-y0-0.1,'$w_{two-phase}$',ha='center',va='center')

#plot color gradient 2nd plot
gradient2 = np.linspace(0, 1, 2)
gradient2 = np.vstack((gradient2, gradient2))
imgplot = pylab.imshow(gradient2, extent=[0.0,4.0,-1.7,-0.71], alpha=0.9, aspect=4, cmap=plt.get_cmap('brg'))
imgplot.set_clim(-0.2,2.0)

pylab.xlim(-0.1,4.3)
pylab.gca().axis('equal')
pylab.gca().axis('off')
pylab.savefig('CondenserSections2.pdf')
pylab.show()
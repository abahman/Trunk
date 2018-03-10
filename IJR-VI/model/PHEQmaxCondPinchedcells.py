import pylab,numpy as np
import CoolProp
from CoolProp.Plots import PropertyPlot
from CoolProp.CoolProp import PropsSI
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

fig=pylab.figure(figsize=(8,4))
ax=fig.add_axes((0.04,0,0.5,1.0))
ax.fill(np.r_[0,3,3,0,0],np.r_[0,0,1,1,0],'lightblue')
ax.fill(np.r_[0,3,3,0,0],np.r_[1,1,2,2,1],'pink')
ax.plot([0.005,0.005],[0,2],'k')
ax.plot([2.995,2.995],[0,2],'k')

ax.plot([2.3,2.3],[1,2],'k')
ax.plot([2.3,2.3],[0,1],'k--')
ax.plot([0.6,0.6],[1,2],'k')
ax.plot([0.6,0.6],[0,1],'k--')
ax.text(1.5,0.5,'Single-Phase',ha='center',va='center')
ax.text(0.3,1.5,'Subcool',ha='center',va='center')
ax.text(0.3,1,'1',bbox=dict(facecolor='w'),ha='center',va='center')
ax.text(1.5,1.5,'Two-Phase',ha='center',va='center')
ax.text(1.5,1,'2',bbox=dict(facecolor='w'),ha='center',va='center')
ax.text(2.65,1.5,'Superheat',ha='center',va='center')
ax.text(2.65,1,'3',bbox=dict(facecolor='w'),ha='center',va='center')


ax.axis('equal')
ax.axis('off')

ax2=fig.add_axes((0.62,0.15,0.35,0.7))
ts_plot = PropertyPlot('R410a', 'Ts',unit_system='KSI',axis=ax2)
ts_plot.calc_isolines(CoolProp.iQ, num=2)
ts_plot.draw() #actually draw isoline
ts_plot.isolines.clear() #stop isoline, to avoid ploting the isoline at the end 
pylab.close()
#
p = PropsSI('P','T',280,'Q',0.0,'R410a')
T1 = np.linspace(270,279.9,100)
T2 = np.linspace(280.11,300,100)
T = np.r_[T1,T2]
s = PropsSI('S','T',T,'P',p,'R410a')/1000.0
ax2.plot(s,T,'r')
ax2.plot([s[0],s[-1]],[270,281],'b')
ax2.set_xlim(0.9,2)
ax2.set_ylim(230,360)
ax2.set_xlabel('Entropy [kJ/kg-K]')
ax2.set_ylabel('Temperature [K]')

pylab.savefig('PHEQmaxCondPinchedcells.pdf')
pylab.show()

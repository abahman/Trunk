import pylab,numpy as np
from math import cos,sin
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

# fig=pylab.figure(figsize=(4,7))
# ax=fig.add_axes((0,0,1,1))

# Slanted lines
N=7
w=0.75
phi=np.pi/3.0
for y0 in np.linspace(0,1,N):
    x=[-w/2.0,0,w/2.0]
    y=[-w/2.0*sin(phi)+y0,y0,-w/2.0*sin(phi)+y0]
    pylab.plot(x,y,'k')

# The frame
x=[w/2.0,w/2.0,-w/2.0,-w/2.0,w/2.0]
y=[-w/2.0*sin(phi),1,1,-w/2.0*sin(phi),-w/2.0*sin(phi)]
pylab.plot(x,y,'k--')

# The inclination angle
pylab.text(0.02,-0.08,r'$\varphi$',ha='left',va='top')
pylab.plot([0,0],[0,-w/2.0*sin(phi)],'k:')

# The ports
t=np.linspace(0,np.pi*2.0,300)
rP=0.125
portsxy=[(-w/2.0+rP,1+rP),(-w/2.0+rP,-w/2.0*sin(phi)-rP),(w/2.0-rP,1+rP),(w/2.0-rP,-w/2.0*sin(phi)-rP)]

for x0,y0 in portsxy:
    pylab.plot(x0+rP*np.cos(t),y0+rP*np.sin(t),'k')
    pylab.fill(x0+rP*np.cos(t),y0+rP*np.sin(t),'grey')

#rounded edges
rE=0.15
y0E=(1.0-w/2.0*sin(phi))/2
wE=1.2*w
hE=2.9*(1.0-w/2.0*sin(phi))

t=np.linspace(0,np.pi/2,100)
x1=wE/2.0-rE+rE*np.cos(t)
y1=hE/2.0-rE+rE*np.sin(t)+y0E
t=np.linspace(np.pi/2,np.pi,100)
x2=-wE/2.0+rE+rE*np.cos(t)
y2=hE/2.0-rE+rE*np.sin(t)+y0E
t=np.linspace(np.pi,3.0*np.pi/2.0,100)
x3=-wE/2.0+rE+rE*np.cos(t)
y3=-hE/2.0+rE+rE*np.sin(t)+y0E
t=np.linspace(3.0*np.pi/2.0,2*np.pi,100)
x4=wE/2.0-rE+rE*np.cos(t)
y4=-hE/2.0+rE+rE*np.sin(t)+y0E
pylab.plot(np.r_[x1,x2,x3,x4,x1[0]],np.r_[y1,y2,y3,y4,y1[0]],'k')

#Length labels
pylab.gca().add_patch(FancyArrowPatch((wE/2.0+0.08,-w/2.0*sin(phi)),(wE/2.0+0.08,1),arrowstyle='<|-|>',fc='k',ec='k',mutation_scale=10,lw=0.5))
pylab.text(wE/2.0+0.1,y0E,'$L$',ha='left',va='center')
pylab.gca().add_patch(FancyArrowPatch((0,y0E+hE/2.0+0.06),(w/2.0,y0E+hE/2.0+0.06),arrowstyle='<|-|>',fc='k',ec='k',mutation_scale=10,lw=0.5))
pylab.text(w/4.0,y0E+hE/2.0+0.08,'$B$',ha='center',va='bottom')

pylab.gca().add_patch(FancyArrowPatch((-wE/2.0-0.08,portsxy[2][1]),(-wE/2.0-0.08,portsxy[3][1]),arrowstyle='<|-|>',fc='k',ec='k',mutation_scale=10,lw=0.5))
pylab.text(-wE/2.0-0.1,y0E,'$L_p$',ha='right',va='center')
pylab.gca().add_patch(FancyArrowPatch((portsxy[3][0],y0E-hE/2.0-0.06),(portsxy[0][0],y0E-hE/2.0-0.06),arrowstyle='<|-|>',fc='k',ec='k',mutation_scale=10,lw=0.5))
pylab.text(0,y0E-hE/2.0-0.08,'$B_p$',ha='center',va='top')

pylab.gca().axis('equal')
pylab.gca().axis('off')
pylab.savefig('PHEOverallGeometry.pdf')
pylab.show()
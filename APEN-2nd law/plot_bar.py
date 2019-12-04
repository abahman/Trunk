import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib as mpl
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

#import data from excel file
df1 = pd.read_excel('data_tables.xlsx',sheet_name='Exergy_des',header=0) #file name
df2 = pd.read_excel('data_tables.xlsx',sheet_name='Exergy_ratio',header=0) #file name
#assign axes
evap = np.array(df1[:]["Evaporator"])*1000
suc = np.array(df1[:]["Suction line"])*1000
comp = np.array(df1[:]["Compressor"])*1000
dis = np.array(df1[:]["Discharge line"])*1000
cond = np.array(df1[:]["Condenser"])*1000
liq = np.array(df1[:]["Liquid line"])*1000
exp = np.array(df1[:]["Expansion valve"])*1000

     
#########################
##### exergy destruction #######
#########################
width = 0.75       # the width of the bars
p1 = plt.bar(np.arange(1,4,1),evap,width,color='k',linewidth=0.9,align='center',alpha=0.9,label=r'Evaporator')
p2 = plt.bar(np.arange(1,4,1),suc,width,bottom=evap,color='b',linewidth=0.9,align='center',alpha=0.9,label=r'Suction line')
p3 = plt.bar(np.arange(1,4,1),comp,width,bottom=[i+j for i,j in zip(evap, suc)],color='r',linewidth=0.9,align='center',alpha=0.9,label=r'Compressor')
p4 = plt.bar(np.arange(1,4,1),dis,width,bottom=[i+j+k for i,j,k in zip(evap, suc, comp)],color='g',linewidth=0.9,align='center',alpha=0.9,label=r'Discharge line')
p5 = plt.bar(np.arange(1,4,1),cond,width,bottom=[i+j+k+l for i,j,k,l in zip(evap, suc, comp, dis)],color='c',linewidth=0.9,align='center',alpha=0.9,label=r'Condenser')
p6 = plt.bar(np.arange(1,4,1),liq,width,bottom=[i+j+k+l+m for i,j,k,l,m in zip(evap, suc, comp, dis, cond)],color='y',linewidth=0.9,align='center',alpha=0.9,label=r'Liquid line')
p7 = plt.bar(np.arange(1,4,1),exp,width,bottom=[i+j+k+l+m+n for i,j,k,l,m,n in zip(evap, suc, comp, dis, cond,liq)],color='m',linewidth=0.9,align='center',alpha=0.9,label=r'Expansion valve')

# def autolabel(rects,p,func):
#     for idx,rect in enumerate(p):
#         height = rect.get_height()
#         plt.text(rect.get_x() + rect.get_width()/2., 0.5*height,
#                 int(func[idx]),
#                 ha='center', va='center', rotation=0,bbox=dict(color='white'), fontsize=8)
# autolabel(p1,p1,evap)
# autolabel(p3,p3,comp)
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(p1)
autolabel(p2)

    

plt.xticks([0, 1, 2, 3, 4],
           [r'',r'1.5-RT', r'3-RT', r'5-RT', r''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
#plt.xlabel(r'Test condition')
plt.ylabel(r'$\dot I$ [W]')
plt.ylim(0,9000)
leg=plt.legend(loc='upper left',scatterpoints=1,scatteryoffsets=[0.5])
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2)        
plt.tick_params(direction='in')
plt.savefig('exergy_des.pdf')
plt.show()
plt.close()
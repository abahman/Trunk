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

def savefigs(name):
    #plt.savefig(name+'.eps')
    plt.savefig(name+'.pdf')
#     plt.savefig(name+'.png',dpi=600)
    #plt.show()
    
################################
##### exergy destruction #######
################################
#import data from excel file
df1 = pd.read_excel('data_tables.xlsx',sheet_name='Exergy_des',header=0) #file name
#exergy destruction
evap = np.array(df1[:]["Evaporator"])*1000
suc = np.array(df1[:]["Suction line"])*1000
comp = np.array(df1[:]["Compressor"])*1000
dis = np.array(df1[:]["Discharge line"])*1000
cond = np.array(df1[:]["Condenser"])*1000
liq = np.array(df1[:]["Liquid line"])*1000
exp = np.array(df1[:]["Expansion valve"])*1000


width = 0.6       # the width of the bars
p7 = plt.bar(np.arange(1,4,1),exp,width,bottom=evap+suc+comp+dis+cond+liq,color='m',linewidth=0.9,align='center',alpha=0.9,label=r'Expansion valve',hatch=2*'\\')
p6 = plt.bar(np.arange(1,4,1),liq,width,bottom=evap+suc+comp+dis+cond,color='y',linewidth=0.9,align='center',alpha=0.9,label=r'Liquid line',hatch=1*'//')
p5 = plt.bar(np.arange(1,4,1),cond,width,bottom=evap+suc+comp+dis,color='c',linewidth=0.9,align='center',alpha=0.9,label=r'Condenser',hatch=2*'x')
p4 = plt.bar(np.arange(1,4,1),dis,width,bottom=evap+suc+comp,color='g',linewidth=0.9,align='center',alpha=0.9,label=r'Discharge line',hatch=2*'+')
p3 = plt.bar(np.arange(1,4,1),comp,width,bottom=evap+suc,color='r',linewidth=0.9,align='center',alpha=0.9,label=r'Compressor',hatch=3*'-')
p2 = plt.bar(np.arange(1,4,1),suc,width,bottom=evap,color='b',linewidth=0.9,align='center',alpha=0.9,label=r'Suction line',hatch=2*'*')
p1 = plt.bar(np.arange(1,4,1),evap,width,color='orange',linewidth=0.9,align='center',alpha=0.9,label=r'Evaporator',hatch=2*'.')
 
def autolabel(p1,p2,p3,p4,p5,p6,p7,func1,func2,func3,func4,func5,func6,func7):
    'function used to plot the values on the stacked bar'
    h = [0,0,0]
    for idx,rect in enumerate(p1):
        height = rect.get_height()
        h[idx] += height
        plt.text(rect.get_x() + rect.get_width()/2., 0.5*h[idx],
                int(func1[idx]),
                ha='center', va='center', rotation=0, fontsize=10,color='white')
    for idx,rect in enumerate(p2):
        height = rect.get_height()
        h[idx] += height
        plt.text(rect.get_x() + rect.get_width()+0.08, h[idx]-height/2,
                int(func2[idx]),
                ha='center', va='center', rotation=0, fontsize=10,color='k')
    for idx,rect in enumerate(p3):
        height = rect.get_height()
        h[idx] += height
        plt.text(rect.get_x() + rect.get_width()/2., h[idx]-height/2,
                int(func3[idx]),
                ha='center', va='center', rotation=0, fontsize=10,color='white')
    for idx,rect in enumerate(p4):
        height = rect.get_height()
        h[idx] += height
        plt.text(rect.get_x() + rect.get_width()+0.08, h[idx]-height/2,
                int(func4[idx]),
                ha='center', va='center', rotation=0, fontsize=10,color='k')
    for idx,rect in enumerate(p5):
        height = rect.get_height()
        h[idx] += height
        plt.text(rect.get_x() + rect.get_width()/2., h[idx]-height/2,
                int(func5[idx]),
                ha='center', va='center', rotation=0, fontsize=10,color='white')
    for idx,rect in enumerate(p6):
        height = rect.get_height()
        h[idx] += height
        plt.text(rect.get_x() + rect.get_width()+0.08, h[idx]-height/2,
                round(func6[idx],1),
                ha='center', va='center', rotation=0, fontsize=10,color='k')
    for idx,rect in enumerate(p7):
        height = rect.get_height()
        h[idx] += height
        plt.text(rect.get_x() + rect.get_width()/2., h[idx]-height/2,
                int(func7[idx]),
                ha='center', va='center', rotation=0, fontsize=10,color='white')    

#plot the numbers
autolabel(p1,p2,p3,p4,p5,p6,p7,evap,suc,comp,dis,cond,liq,exp)

    
plt.xticks([1, 2, 3],
           [r'1.5-RT', r'3-RT', r'5-RT'])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
#plt.xlabel(r'Test condition')
plt.ylabel(r'$\dot I$ (W)')
plt.ylim(0,8000)
# leg=plt.legend(loc='upper left',scatterpoints=1,scatteryoffsets=[0.5])
leg = plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.22),fancybox=False,numpoints=1,ncol=3)#, bbox_to_anchor=(0.5, 1.15)
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2)        
plt.tick_params(direction='in')
savefigs('irrev2')
plt.show()
plt.close()



#########################
##### exergy destruction ratio#######
#########################
#import data from excel file
df2 = pd.read_excel('data_tables.xlsx',sheet_name='Exergy_ratio',header=0) #file name
#exergy destruction ration
evap = np.array(df2[:]["Evaporator"])
suc = np.array(df2[:]["Suction line"])
comp = np.array(df2[:]["Compressor"])
dis = np.array(df2[:]["Discharge line"])
cond = np.array(df2[:]["Condenser"])
liq = np.array(df2[:]["Liquid line"])
exp = np.array(df2[:]["Expansion valve"])


width = 0.6       # the width of the bars
p7 = plt.bar(np.arange(1,4,1),exp,width,bottom=evap+suc+comp+dis+cond+liq,color='m',linewidth=0.9,align='center',alpha=0.9,label=r'Expansion valve',hatch=2*'\\')
p6 = plt.bar(np.arange(1,4,1),liq,width,bottom=evap+suc+comp+dis+cond,color='y',linewidth=0.9,align='center',alpha=0.9,label=r'Liquid line',hatch=1*'//')
p5 = plt.bar(np.arange(1,4,1),cond,width,bottom=evap+suc+comp+dis,color='c',linewidth=0.9,align='center',alpha=0.9,label=r'Condenser',hatch=2*'x')
p4 = plt.bar(np.arange(1,4,1),dis,width,bottom=evap+suc+comp,color='g',linewidth=0.9,align='center',alpha=0.9,label=r'Discharge line',hatch=2*'+')
p3 = plt.bar(np.arange(1,4,1),comp,width,bottom=evap+suc,color='r',linewidth=0.9,align='center',alpha=0.9,label=r'Compressor',hatch=3*'-')
p2 = plt.bar(np.arange(1,4,1),suc,width,bottom=evap,color='b',linewidth=0.9,align='center',alpha=0.9,label=r'Suction line',hatch=2*'*')
p1 = plt.bar(np.arange(1,4,1),evap,width,color='orange',linewidth=0.9,align='center',alpha=0.9,label=r'Evaporator',hatch=2*'.')
 
def autolabel_new(p1,p2,p3,p4,p5,p6,p7,func1,func2,func3,func4,func5,func6,func7):
    'function used to plot the values on the stacked bar'
    h = [0,0,0]
    for idx,rect in enumerate(p1):
        height = rect.get_height()
        h[idx] += height
        plt.text(rect.get_x() + rect.get_width()/2., 0.5*h[idx],
                round(func1[idx],1),
                ha='center', va='center', rotation=0, fontsize=10,color='white')
    for idx,rect in enumerate(p2):
        height = rect.get_height()
        h[idx] += height
        plt.text(rect.get_x() + rect.get_width()+0.08, h[idx]-height/2,
                round(func2[idx],1),
                ha='center', va='center', rotation=0, fontsize=10,color='k')
    for idx,rect in enumerate(p3):
        height = rect.get_height()
        h[idx] += height
        plt.text(rect.get_x() + rect.get_width()/2., h[idx]-height/2,
                round(func3[idx],1),
                ha='center', va='center', rotation=0, fontsize=10,color='white')
    for idx,rect in enumerate(p4):
        height = rect.get_height()
        h[idx] += height
        plt.text(rect.get_x() + rect.get_width()+0.08, h[idx]-height/2,
                round(func4[idx],1),
                ha='center', va='center', rotation=0, fontsize=10,color='k')
    for idx,rect in enumerate(p5):
        height = rect.get_height()
        h[idx] += height
        plt.text(rect.get_x() + rect.get_width()/2., h[idx]-height/2,
                round(func5[idx],1),
                ha='center', va='center', rotation=0, fontsize=10,color='white')
    for idx,rect in enumerate(p6):
        height = rect.get_height()
        h[idx] += height
        plt.text(rect.get_x() + rect.get_width()+0.1, h[idx]-height/2,
                round(func6[idx],2),
                ha='center', va='center', rotation=0, fontsize=10,color='k')
    for idx,rect in enumerate(p7):
        height = rect.get_height()
        h[idx] += height
        plt.text(rect.get_x() + rect.get_width()/2., h[idx]-height/2,
                round(func7[idx],1),
                ha='center', va='center', rotation=0, fontsize=10,color='white')    

#plot the numbers
autolabel_new(p1,p2,p3,p4,p5,p6,p7,evap,suc,comp,dis,cond,liq,exp)

    
plt.xticks([1, 2, 3],
           [r'1.5-RT', r'3-RT', r'5-RT'])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
#plt.xlabel(r'Test condition')
plt.ylabel(r'$E_d$ ($\%$)')
plt.ylim(0,100)
leg = plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.22),fancybox=False,numpoints=1,ncol=3)
# leg=plt.legend(loc='upper left',scatterpoints=1,scatteryoffsets=[0.5])
frame=leg.get_frame()  
frame.set_linewidth(0.5)
plt.tight_layout(pad=0.2)        
plt.tick_params(direction='in')
savefigs('exergy_ratio2')
plt.show()
plt.close()
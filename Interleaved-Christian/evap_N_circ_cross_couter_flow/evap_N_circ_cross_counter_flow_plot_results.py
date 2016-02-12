import pylab as plt
import numpy as np

import  pandas as pd
import sys

sys.path.append("C:/Users/bachc/Desktop/achp/trunk/PyACHP") #for pandas tools
from PandasTools import get_values #custom function

def plot_these(df_nonhybrid,df_hybrid,filename_save,legend_title,normalize=False):
    def plot_this(sourceframe,conf_type,Name,symbol=None):
        index_this=sourceframe.ix[sourceframe['Description']==conf_type]  #get indexes
        print conf_type
        Q=np.array(get_values(sourceframe,index_this.index,"Q Total"))
        Q=Q.astype(np.float) #need to convert to float
        if normalize:
            Q/=Q.max()
        Epsilon=np.array(get_values(sourceframe,index_this.index,"Epsilon_md")).astype(np.float)
        if symbol:
            plt.plot(Epsilon,Q,'ro',ls='-',c='k',label=Name)
        else:
            plt.plot(Epsilon,Q,label=Name)
        return Epsilon.max()
        
    fig, ax1 = plt.subplots()
    
    for conf_type in ['Standard','Interleaved','Equal flow']:
        Name=conf_type
        if conf_type=='Standard':
            plot_this(df_nonhybrid,conf_type,Name,symbol='ro')
        else:
            x_max=plot_this(df_nonhybrid,conf_type,Name)
    #plot hybrid cases 
    plot_this(df_hybrid,'Standard','Hybrid')
    #add axis
    if normalize: 
        plt.ylim(0,1.05)
    else:
        plt.ylim(0,14000)
    
    plt.xlim(0,x_max*1.05)
    plt.rcParams['font.size']=18
    plt.legend(loc='best',fancybox=True,title=legend_title)
    if normalize:
        ax1.set_ylabel('Normalized Capacity')
    else:
        ax1.set_ylabel('Capacity in [W]')
    ax1.set_xlabel('Dimensionless Maldistribution')
    #remove first y label, since we have some overlap...
    if False:
        if ax1.yaxis.get_majorticklocs()[-1]>10:
            labels = [str(int(item)) for item in ax1.yaxis.get_majorticklocs()]
        else:
            labels = [str(item) for item in ax1.yaxis.get_majorticklocs()]
        labels[0] = labels[0]+'\n'
        ax1.set_yticklabels(labels)
    else:
        labels = [str(item) for item in ax1.xaxis.get_majorticklocs()]
        labels[0] = '0'
        ax1.set_xticklabels(labels)
    
    #fig.subplots_adjust(left=0.17,bottom=0.15,right=0.95)
    plt.savefig(filename_save,dpi=600,bbox_inches='tight')
    
base_folder_nonhybrid='D:/Purdue/DOE Project/Simulation/LRCS/noniter-cases/'
basefolder_hybrid='D:/Purdue/DOE Project/Simulation/LRCS/iternum=10,integ=0.07/'
basefolder_save_figure='D:/Purdue/DOE Project/Simulation/LRCS/png_out_with_equalflow/'

filenames_hybrid=['sh_equalizer_tester_adjust_superheat_iter_Halflinear A.csv','sh_equalizer_tester_adjust_superheat_iter_Halflinear B.csv','sh_equalizer_tester_adjust_superheat_iter_linear.csv','sh_equalizer_tester_adjust_superheat_iter_pyramid.csv']
filenames_nonhybrid=['Halflinear ALRCS.csv','Halflinear BLRCS.csv','linearLRCS.csv','pyramidLRCS.csv']
legend_titles=['Halflinear A', 'Halflinear B', 'Linear','Pyramid']#png_out filenames are generated from legendtitle

filenamegroups=[]
for i in range(len(filenames_hybrid)):
    filenamegroups.append([base_folder_nonhybrid+filenames_nonhybrid[i],basefolder_hybrid+filenames_hybrid[i],basefolder_save_figure+legend_titles[i],legend_titles[i]])

#capacity plots
for filenamegroup in filenamegroups:
    df_nonhybrid=pd.read_csv(filenamegroup[0],skiprows=1,index_col=1)
    df_hybrid=pd.read_csv(filenamegroup[1],skiprows=1,index_col=1)
    filename_save=filenamegroup[2]
    legend_title=filenamegroup[3]
    plot_these(df_nonhybrid,df_hybrid,filename_save,legend_title)
    plot_these(df_nonhybrid,df_hybrid,filename_save+'Norm',legend_title,normalize=True)
plt.show()

#rel capacity plots
#rel. capacity versus maldistribution plot

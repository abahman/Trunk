import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def select_stringbased(df_in,header_y,header_x,header_sel,val_sel_is):
    #return a data series out of df_in for header_x and header_y, select based on header_sel
    #which should be equal to val_sel_is
    return pd.Series(df_in[header_y][df_in[header_sel]==val_sel_is], df_in[header_x][df_in[header_sel]==val_sel_is])

def read_plot(Filename_in,header_ys,header_x,header_sel,val_sel_list,Left=0.12, Bottom=0.13, Right=0.95, Top=0.95,y_lims=[0.0,1.1],Normalize_y=True):
    #for comparison between different configurations
    #read in data_file, plot each series in label list
    #make a plot for everything in columns_list
    #normalise whatever is set to true in normalise list
    
    df_in=pd.read_csv(Filename_in,skiprows=[0,2])

    for header_y in header_ys:
        if Normalize_y:
            Factor_y=1.0/max(df_in[header_y][df_in[header_sel]==val_sel_list[0]])
        else:
             Factor_y=1.0
    
        plt.figure(figsize=(6*0.8, 4.5*0.8))
        for i in range(len(val_sel_list)):
            tmp=select_stringbased(df_in,header_y,header_x,header_sel,val_sel_list[i])*Factor_y
            #print i,val_sel_list[i],tmp
            tmp.plot(val_sel_list[i])
            print tmp
        plt.xlabel(header_x)
        
    if Normalize_y:
        plt.ylabel('Normalized '+header_y)
    else:
        plt.ylabel(header_y)
    plt.ylim(y_lims)
    leg=plt.legend(loc='best',fancybox=True,ncol=1)
    leg.get_frame().set_alpha(0.3)
    plt.subplots_adjust(left=Left, bottom=Bottom, right=Right, top=Top)#relative measure
    Filename=Filename_in[:-4]+'.png'
    plt.savefig(Filename, dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=True, bbox_inches=None, pad_inches=0.2)
    return

def read_plot2(Filename_in,header_ys,header_x,header_sel,val_sel_list,Left=0.12, Bottom=0.13, Right=0.95, Top=0.95,y_lims=[None,None],Normalize_y=True):
    #for comparison between different values for each singele configuration, e.g. SH-plots
    #read in data_file, plot each series in label list
    #make a plot for everything in columns_list
    #normalise whatever is set to true in normalise list
    
    df_in=pd.read_csv(Filename_in,skiprows=[0,2])
    
    for i in range(len(val_sel_list)):
        plt.figure(figsize=(6*0.8, 4.5*0.8))
        for i_h in range(len(header_ys)):
            header_y=header_ys[i_h]
            if Normalize_y:
                Factor_y=1.0/max(df_in[header_y][df_in[header_sel]==val_sel_list[0]])
            else:
                 Factor_y=1.0
            tmp=select_stringbased(df_in,header_y,header_x,header_sel,val_sel_list[i])*Factor_y
            tmp.plot(str(i_h))
            plt.xlabel(header_x)
        if Normalize_y:
            plt.ylabel('Normalized '+header_y[:-4])
        else:
            plt.ylabel(header_y[:-4] + " " + header_y[-3:-2])
        plt.ylim(y_lims)
        leg=plt.legend(loc='best',fancybox=True,ncol=2)
        leg.get_frame().set_alpha(0.3)
        plt.subplots_adjust(left=Left, bottom=Bottom, right=Right, top=Top)#relative measure
        Filename=Filename_in[:-4]+header_ys[i_h]+val_sel_list[i]+'.png'
        plt.savefig(Filename, dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=True, bbox_inches=None, pad_inches=0.2)
    return

if __name__=='__main__':
    
    headers_y=[['Outlet superheat_A ']*8,['Wetted Area Fraction Superheat_A ']*8]
    for i_h in range(len(headers_y)):
        for i in range(len(headers_y[i_h])):
            headers_y[i_h][i]=headers_y[i_h][i]+str(i)
            
    
    header_x='Maldistribution Severity'
    header_sel='Description'
    val_sel_list=['Standard','Interleaved','Equal flow']
    #for airside and combined maldistribtion
    #Filenames=['RAC-NCircuit_airMD_avg.csv','RAC-NCircuit_airTMD_typeRAC_Temp.csv','RAC-NCircuit_combined_RTU_MD.csv','LRCS-NCircuit_airMD_typeA.csv']
    import os
    print "current working directory",os.getcwd()
    Filenames=[os.getcwd()+"\\Ncircuit_sims2\\"+'LRCS-NCircuit_airMD_typeA.csv']
    if 0:
        for Filename_in in Filenames:
            for header_y in headers_y:
                print "working on ", Filename_in
                if header_y[0]=='Outlet superheat_A 0':
                    Y_lims=[-2,12]
                elif header_y[0]=='Wetted Area Fraction Superheat_A 0':
                    Y_lims=[0,1.1]
                read_plot2(Filename_in,header_y,header_x,header_sel,val_sel_list,Normalize_y=False,y_lims=Y_lims)
    
    headers_y=[['Outlet superheat_A ']*7,['Wetted Area Fraction Superheat_A ']*7]
    for i_h in range(len(headers_y)):
        for i in range(len(headers_y[i_h])):
            headers_y[i_h][i]=headers_y[i_h][i]+str(i)
    Filenames=[os.getcwd()+"\\Ncircuit_sims2\\"+'RAC-NCircuit_combined_RTU_MD.csv',os.getcwd()+"\\Ncircuit_sims2\\"+'RAC-NCircuit_combined_RTU_MD.csv',os.getcwd()+"\\Ncircuit_sims2\\"+'RAC-NCircuit_airTMD_typeRAC_Temp.csv']
    if 0:
        for Filename_in in Filenames:
            for header_y in headers_y:
                print "working on ", Filename_in
                if header_y[0]=='Outlet superheat_A 0':
                    Y_lims=[-2,12]
                elif header_y[0]=='Wetted Area Fraction Superheat_A 0':
                    Y_lims=[0,1.1]
                read_plot2(Filename_in,header_y,header_x,header_sel,val_sel_list,Normalize_y=False,y_lims=Y_lims)
    
    
    if 0:
        #below for refrigerant side maldistributed cases
        header_y=['Q Total']
        val_sel_list=['Standard','Interleaved']
        Filenames=['RAC-NCircuit_refMD_linear.csv','LRCS-NCircuit_refMD_linear.csv']
        for Filename_in in Filenames:
            print "working on ", Filename_in
            Filename_in='D:\Purdue\DOE Project\Simulation\Ncircuit_sims2'+'\\'+Filename_in
            read_plot(Filename_in,header_y,header_x,header_sel,val_sel_list)
                #below for refrigerant side maldistributed cases
    if 1:
        #below for airside and combined maldistribution
        header_y=['Q Total']
        val_sel_list=['Standard','Interleaved','Equal flow']
        Filenames=['LRCS-NCircuit_airMD_typeA.csv','RAC-NCircuit_airTMD_typeRAC_Temp.csv','RAC-NCircuit_airMD_avg.csv','RAC-NCircuit_combined_RTU_MD.csv']
        for Filename_in in Filenames:
            print "working on ", Filename_in
            Filename_in='D:\Purdue\DOE Project\Simulation\Ncircuit_sims2'+'\\'+Filename_in
            read_plot(Filename_in,header_y,header_x,header_sel,val_sel_list)

    plt.show()

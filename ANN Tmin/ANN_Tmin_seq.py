import os,sys
#os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
import math
from math import log, pi
import xlrd
import time
import warnings
import pylab
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.mlab as ml
from scipy import optimize,stats
import pandas as pd
from openpyxl import load_workbook

warnings.simplefilter("ignore",RuntimeWarning)
from random import randint, random

import DataIO
from CoolProp.CoolProp import PropsSI

plt.style.use('Elsevier.mplstyle')
mpl.style.use('classic')
mpl.style.use('Elsevier.mplstyle')
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['figure.figsize'] = [6,4]
mpl.rcParams['legend.numpoints'] = 1

#--------------------------------------------------------------------------
def Import(start,end,filename):
    "import experimental data"
    
    [data,rownum] = DataIO.ParameterImport(start,end,filename)
    
    i = 0  
    "initialize arrays"
    Tmin = float(data[i][0]) #Tamb = float(data[i][0])
    Tsub = float(data[i][1]) #Tsuc = float(data[i][1])
    Psat = float(data[i][2]) #Pev = float(data[i][2])
    Mat = str(data[i][3]) #Tex = float(data[i][3])
    LD = float(data[i][4]) #Pcd = float(data[i][4])
    #Tinj = float(data[i][5])
    i=i+1
    
    while i < (end - start+1):
        Tmin = np.append(Tmin,float(data[i][0])) #Tamb = np.append(Tamb,float(data[i][0]))
        Tsub = np.append(Tsub,float(data[i][1])) #Tsuc = np.append(Tsuc,float(data[i][1]))
        Psat = np.append(Psat,float(data[i][2])) #Pev = np.append(Pev,float(data[i][2]))
        Mat = np.append(Mat,str(data[i][3])) #Tex = np.append(Tex,float(data[i][3]))
        LD = np.append(LD,float(data[i][4])) #Pcd = np.append(Pcd,float(data[i][4]))
        #Tinj = np.append(Tinj,float(data[i][5]))
        i=i+1
        Data = [Tmin,Tsub,Psat,Mat,LD]
    
    return Data
    
def rmse(predictions, targets):
    '''
    Root Mean Square Error
    '''
    n = len(predictions)
    RMSE = np.linalg.norm(predictions - targets) / np.sqrt(n) / np.mean(targets) * 100
    return RMSE

def mape(y_pred, y_true):  #maps==mean_absolute_percentage_error
    '''
    Mean Absolute Percentage Error
    '''
    MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return MAPE

def mse(y_pred, y_true):
    '''
    Mean Squared Error
    '''
    from sklearn.metrics import mean_squared_error
    MSE = mean_squared_error(y_pred, y_true)
    return MSE

def Normalize(y_data,y_data_min,y_data_max):
    
    y_norm = 0.8*(y_data - y_data_min)/(y_data_max - y_data_min) + 0.1
    
    return y_norm

def DeNormalize(y_norm,y_data_min,y_data_max):
    
    y = (y_norm - 0.1)*(y_data_max - y_data_min)/0.8 + y_data_min
    
    return y

def REmean(y_true,y_pred):
    
    return np.mean(np.fabs(y_true - y_pred)/y_true)    

def Rsquared(y_true,y_pred):
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true,y_pred)    
    
    return r_value**2
    
def Calculate():

    "Import Experimental Data"
    start=1
    end=376
    filename = 'Data_Collection.csv' #'scroll_inj_R407C.csv'
    
    #Define inputs
    [Tmin_exp,Tsub,Psat,Mat,LD] = Import(start,end,filename)
    
    mode = 'training'
    
    from keras.models import Model,Sequential,load_model
    from keras.layers import Input,Flatten,Dense,LSTM,merge,Dropout,concatenate
    #from keras.engine import merge # from Keras version 1.2.2
    #from keras.layers.merge import concatenate
    from keras.utils import plot_model
    from keras.callbacks import TensorBoard
    from keras import regularizers
    
    
    #Normalize all parameters
    Tmin_exp_norm = Normalize(Tmin_exp, 206.8841, 727.8873239) #P_ev_norm = Normalize(P_ev,100, 900)
    Tsub_norm = Normalize(Tsub, 0, 70) #T_suc_norm = Normalize(T_suc,263.15,300)
    Psat_norm = Normalize(Psat, 0.001185867, 3.003378378) #P_cd_norm = Normalize(P_cd,1000, 3500)
    LD_norm = Normalize(LD, 2.67, 63.5) #T_inj_norm = Normalize(T_inj,273.15,330.15)

    Tmin_exp_norm = np.array(Tmin_exp_norm)
    Tsub_norm = np.array(Tsub_norm)
    Psat_norm = np.array(Psat_norm)
    LD_norm = np.array(LD_norm)
        
    if mode == 'training':

        # split into input (X) and output (Y) variables
        X = np.column_stack((Tsub_norm, Psat_norm))
        X = np.column_stack((X, LD_norm))
        Y = Tmin_exp_norm


        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=3, activation='tanh')) #init='uniform'
        model.add(Dense(12, activation='tanh'))
        model.add(Dense(1, activation='linear'))
          
        plot_model(model, to_file='model.pdf',show_shapes=True,show_layer_names=True)
  
        # Compile model
        model.compile(optimizer='adamax',loss='mse',metrics=['mae'])
          
        # fit the model
        history = model.fit(X,
                            Y,
                            epochs=4000 ,
                            batch_size=30, #increase the batch size results in faster compiler an d high error, while smaller batch size results in slower compiler and slightly accurate model
                            validation_split=0.2,
                            )    
          
        # evaluate the model
        scores = model.evaluate(X,Y)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            
    #   #History plot
        fig=pylab.figure(figsize=(6,4))
        plt.semilogy(history.history['loss'])
        plt.semilogy(history.history['val_loss'])
        #plt.semilogy(history.history['val_mean_absolute_error'])
        plt.ylabel('loss [-]')
        plt.xlabel('epoch [-]')
        plt.legend(['Train', 'Test'], loc='upper right',fontsize=9)
        #plt.ylim(0,0.1)
        plt.tight_layout(pad=0.2)  
        plt.tick_params(direction='in')      
        fig.savefig('ANN_history_Tmin.pdf')
        
        # Save the model
        model.save('ANN_model_Tmin.h5')
    
    elif mode == 'run':
    
        # Load the model
        model = load_model('ANN_model_Tmin.h5')
    
    # Run the model
    Tmin_ANN = model.predict(X) #[Mref,Minj,W,T,eta_s,Q] = model.predict([P_ev_norm,T_suc_norm,P_cd_norm,T_inj_norm,P_inj_norm,T_amb_norm])
    Tmin_ANN = DeNormalize(Tmin_ANN.reshape(-1), 206.8841, 727.8873239) #W = DeNormalize(W.reshape(-1),1000,8000)


    # Save the architecture of a model, and not its weights or its training configuration
    # save as JSON
    # json_string = model.to_json()
    
    # save as YAML
    # yaml_string = model.to_yaml()


    for i in range(0,(end-start+1)):


        data_calc = {'Tmin':[Tmin_ANN[i]]}  #data_calc = {'Tdis':[T[i]],'mdot':[Mref[i]],'mdot_inj':[Minj[i]], 'Wdot':[W[i]],'etaoa':[eta_s[i]],'fq':[Q[i]/W[i]]} 
            
        
        # Write to Excel
        filename = os.path.dirname(__file__)+'/Tmin_output.xlsx'
        xl = pd.read_excel(filename, sheet_name='ANN_Validation')

        df = pd.DataFrame(data=data_calc)

        df.reindex(columns=xl.columns)
        df_final=xl.append(df,ignore_index=True)
        df_final.tail()
        
        book = load_workbook(filename)
        writer = pd.ExcelWriter(filename, engine='openpyxl',index=False)
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        df_final.to_excel(writer,index=False,sheet_name='ANN_Validation')
        
        # 
        writer.save()

    
    #Separate testing from calibrating
    sep_val = 0.2
    n_len = len(Tmin_ANN)
    n_split = int(np.floor(sep_val*n_len))
    n_training = int(n_len-n_split-1)


    # Validation Tmin
    fig=pylab.figure(figsize=(4,4))

    plt.plot(Tmin_ANN[:n_training],Tmin_exp[:n_training],'ro',ms = 3,mec='black',mew=0.5,label='Training points')
    plt.plot(Tmin_ANN[-n_split:],Tmin_exp[-n_split:],'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
    plt.text(550,200,'R$^2$ = {:0.01f}%\n'.format(Rsquared(Tmin_exp,Tmin_ANN)*100)+'MAE = {:0.01f}%\n'.format(mape(Tmin_ANN,Tmin_exp))+'RMSE = {:0.01f}%'.format(rmse(Tmin_ANN,Tmin_exp)),ha='left',va='center',fontsize = 8)

    plt.xlabel('$T_{min,pred}$ [$\degree$C]')
    plt.ylabel('$T_{min,exp}$ [$\degree$C]')

    Tmin = 100
    Tmax = 800
    x=[Tmin,Tmax]
    y=[Tmin,Tmax]
    y105=[1.1*Tmin,1.1*Tmax]
    y95=[0.9*Tmin,0.9*Tmax]
    
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)    
    plt.xlim(Tmin,Tmax)
    plt.ylim(Tmin,Tmax)
    plt.legend(loc=2,fontsize=9)
    plt.tight_layout(pad=0.2)        
    plt.tick_params(direction='in')
    plt.show()
    fig.savefig('ANN_Tmin.pdf')
    
    print 'Tmin:',REmean(Tmin_exp,Tmin_ANN),Rsquared(Tmin_exp,Tmin_ANN)*100 #print 'Wdot:',REmean(W_meas,W),Rsquared(W_meas,W)*100


    
if __name__ == '__main__':
    
    Calculate()
    
    """
    Wdot: 0.008220341979717799 99.56922288736178
    mdot: 0.011331759697160361 99.64595206448986
    minj: 0.07011354008261375 96.63843422122608
    Tex: 0.002574871815676044 98.65646548151139
    eta_s: 0.009537828936222303 79.76583704167605
    f_q: 0.07263509578119003 92.44388296110533
    m_inj/m_suc: 0.07515435340478782 96.01705488251565
    """
    
    
    
    
    
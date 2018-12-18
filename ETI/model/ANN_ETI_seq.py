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
    T_env = float(data[i][0])
    T_suc = float(data[i][1])
    P_suc = float(data[i][2])
    T_dis = float(data[i][3])
    P_dis = float(data[i][4])
    T_inj = float(data[i][5])
    P_inj = float(data[i][6])
    x_inj = float(data[i][7])
    h_inj = float(data[i][8])
    m_suc = float(data[i][9])
    m_inj = float(data[i][10])
    m_tot = float(data[i][11])
    Q_loss = float(data[i][12])
    W_comp = float(data[i][13])
    f_loss = float(data[i][14])
    i=i+1
    
    while i < (end - start+1):
        T_env = np.append(T_env,float(data[i][0]))
        T_suc = np.append(T_suc,float(data[i][1]))
        P_suc = np.append(P_suc,float(data[i][2])) 
        T_dis = np.append(T_dis,float(data[i][3]))
        P_dis = np.append(P_dis,float(data[i][4]))
        T_inj = np.append(T_inj,float(data[i][5]))
        P_inj = np.append(P_inj,float(data[i][6]))
        x_inj = np.append(x_inj,float(data[i][7]))
        h_inj = np.append(h_inj,float(data[i][8]))
        m_suc = np.append(m_suc,float(data[i][9]))
        m_inj = np.append(m_inj,float(data[i][10]))
        m_tot = np.append(m_tot,float(data[i][11]))
        Q_loss = np.append(Q_loss,float(data[i][12]))
        W_comp = np.append(W_comp,float(data[i][13]))
        f_loss = np.append(f_loss,float(data[i][14]))
        i=i+1
        Data = [T_env,T_suc,P_suc,T_dis,P_dis,T_inj,P_inj,x_inj,h_inj,m_suc,m_inj,m_tot,Q_loss,W_comp,f_loss]
    
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
    end=67
    filename = 'data_ETI.csv'
    
    #Define inputs
    [T_env,T_suc,P_suc,T_dis,P_dis,T_inj,P_inj,x_inj,h_inj,m_suc,m_inj,m_tot,Q_loss,W_comp,f_loss] = Import(start,end,filename)
    
    mode = 'run'
    
    from keras.models import Model,Sequential,load_model
    from keras.layers import Input,Flatten,Dense,LSTM,merge,Dropout,concatenate
    #from keras.engine import merge # from Keras version 1.2.2
    #from keras.layers.merge import concatenate
    from keras.utils import plot_model
    from keras.callbacks import TensorBoard
    from keras import regularizers
    
    
    #Normalize all parameters
    T_env_norm = Normalize(T_env, 297, 324.8)
    T_suc_norm = Normalize(T_suc, 274.1, 294.53)
    P_suc_norm = Normalize(P_suc, 327, 749.6)
    T_dis_norm = Normalize(T_dis, 342.5, 380.5)
    P_dis_norm = Normalize(P_dis, 1434, 3186)
    #T_inj_norm = Normalize(T_inj, 279.8, 322.5)
    P_inj_norm = Normalize(P_inj, 552.9, 1588)
    h_inj_norm = Normalize(h_inj, 378.6, 448.7)
    m_suc_norm = Normalize(m_suc, 0.0466, 0.1045)
    m_inj_norm = Normalize(m_inj, 0.00496, 0.04302)
    m_tot_norm = Normalize(m_tot, 0.05507, 0.14702)
    #Q_loss_norm = Normalize(Q_loss, -0.04526, 0.3732792)
    W_comp_norm = Normalize(W_comp, 3.154, 7.806)
    f_loss_norm = Normalize(f_loss, -0.8931, 6.691)
    
    #convert to numpy array
    T_env_norm = np.array(T_env_norm)
    T_suc_norm = np.array(T_suc_norm)
    P_suc_norm = np.array(P_suc_norm)
    T_dis_norm = np.array(T_dis_norm)
    P_dis_norm = np.array(P_dis_norm)
    #T_inj_norm = np.array(T_inj_norm)
    P_inj_norm = np.array(P_inj_norm)
    h_inj_norm = np.array(h_inj_norm)
    m_suc_norm = np.array(m_suc_norm)
    m_inj_norm = np.array(m_inj_norm)
    m_tot_norm = np.array(m_tot_norm)
    #Q_loss_norm = np.array(Q_loss_norm)
    W_comp_norm = np.array(W_comp_norm)
    f_loss_norm = np.array(f_loss_norm)
    
    # split into input (X) and output (Y) variables
    X = np.column_stack((T_env_norm, T_suc_norm))
    X = np.column_stack((X, P_suc_norm))
    X = np.column_stack((X, P_dis_norm))
    X = np.column_stack((X, P_inj_norm))
    X = np.column_stack((X, h_inj_norm))
    
    Y = np.column_stack((T_dis_norm,m_suc_norm))
    Y = np.column_stack((Y, m_inj_norm))
    Y = np.column_stack((Y, m_tot_norm))
    Y = np.column_stack((Y, W_comp_norm))
    Y = np.column_stack((Y, f_loss_norm))
    
    if mode == 'training':
        # create model
        model = Sequential()
        model.add(Dense(18, input_dim=6, activation='tanh')) #init='uniform' #use_bias = True, bias_initializer='zero'
        #model.add(Dropout(0.2)) #Dropout is a technique where randomly selected neurons are ignored during training.
        #model.add(Dense(12, activation='tanh'))
        #model.add(Dense(12, activation='tanh'))
        model.add(Dense(6, activation='linear'))
          
        plot_model(model, to_file='model.pdf',show_shapes=True,show_layer_names=True)
  
        # Compile model
        model.compile(optimizer='adamax',loss='mse',metrics=['mae'])
          
        # fit the model
        history = model.fit(X,
                            Y,
                            epochs=8000 , #Cut the epochs in half when using sequential 
                            batch_size=30, #increase the batch size results in faster compiler an d high error, while smaller batch size results in slower compiler and slightly accurate model
                            validation_split=0.2,
                            )    
          
        
            
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
        fig.savefig('ANN_history_ETI.pdf')
        
        # Save the model
        model.save('ANN_model_ETI.h5')
    
    elif mode == 'run':
    
        # Load the model
        model = load_model('ANN_model_ETI.h5')
    
    # Run the model
    predictions = model.predict(X)
    T_dis_ANN = DeNormalize(predictions[:,0].reshape(-1), 342.5, 380.5)
    m_suc_ANN = DeNormalize(predictions[:,1].reshape(-1), 0.0466, 0.1045)
    m_inj_ANN = DeNormalize(predictions[:,2].reshape(-1), 0.00496, 0.04302)
    m_tot_ANN = DeNormalize(predictions[:,3].reshape(-1), 0.05507, 0.14702)
    W_comp_ANN = DeNormalize(predictions[:,4].reshape(-1), 3.154, 7.806)
    f_loss_ANN = DeNormalize(predictions[:,5].reshape(-1), -0.8931, 6.691)

    # evaluate the model
    scores = model.evaluate(X,Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        
    # extract the weight and bias
    weights = model.layers[0].get_weights()[0]
    biases = model.layers[0].get_weights()[1]
    
    print 'weights = ', weights
    print 'biases = ', biases
    # Save the architecture of a model, and not its weights or its training configuration
    # save as JSON
    # json_string = model.to_json()
    
    # save as YAML
    # yaml_string = model.to_yaml()


    for i in range(0,(end-start+1)):


        data_calc = {'T_dis':[T_dis_ANN[i]],'m_suc':[m_suc_ANN[i]],'m_inj':[m_inj_ANN[i]],'m_tot':[m_tot_ANN[i]],'W_comp':[W_comp_ANN[i]],'f_loss':[f_loss_ANN[i]]}
            
        
        # Write to Excel
        filename = os.path.dirname(__file__)+'/ETI_output.xlsx'
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
    n_len = len(T_dis_ANN)
    n_split = int(np.floor(sep_val*n_len))
    n_training = int(n_len-n_split-1)


    # Validation T_dis
    fig=pylab.figure(figsize=(4,4))

    plt.plot(T_dis_ANN[:n_training],T_dis[:n_training],'ro',ms = 3,mec='black',mew=0.5,label='Training points')
    plt.plot(T_dis_ANN[-n_split:],T_dis[-n_split:],'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
    plt.text(370,330,'R$^2$ = {:0.01f}%\n'.format(Rsquared(T_dis,T_dis_ANN)*100)+'MAE = {:0.01f}%\n'.format(mape(T_dis_ANN,T_dis))+'RMSE = {:0.01f}%\n'.format(rmse(T_dis_ANN,T_dis)),ha='left',va='center',fontsize = 8)

    plt.xlabel('$T_{dis,pred}$ [K]')
    plt.ylabel('$T_{dis,exp}$ [K]')

    Tmin = 320
    Tmax = 400
    x=[Tmin,Tmax]
    y=[Tmin,Tmax]
    y105=[1.05*Tmin,1.05*Tmax]
    y95=[0.95*Tmin,0.95*Tmax]
    
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)    
    plt.xlim(Tmin,Tmax)
    plt.ylim(Tmin,Tmax)
    plt.legend(loc=2,fontsize=9)
    plt.tight_layout(pad=0.2)        
    plt.tick_params(direction='in')
    plt.show()
    fig.savefig('ANN_T_dis.pdf')
    
    # Validation m_suc
    fig=pylab.figure(figsize=(4,4))

    plt.plot(m_suc_ANN[:n_training],m_suc[:n_training],'ro',ms = 3,mec='black',mew=0.5,label='Training points')
    plt.plot(m_suc_ANN[-n_split:],m_suc[-n_split:],'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
    plt.text(0.08,0.02,'R$^2$ = {:0.01f}%\n'.format(Rsquared(m_suc,m_suc_ANN)*100)+'MAE = {:0.01f}%\n'.format(mape(m_suc_ANN,m_suc))+'RMSE = {:0.01f}%\n'.format(rmse(m_suc_ANN,m_suc)),ha='left',va='center',fontsize = 8)

    plt.xlabel('$\dot m_{suc,pred}$ [kg/s]')
    plt.ylabel('$\dot m_{suc,exp}$ [kg/s]')

    Tmin = 0.0
    Tmax = 0.12
    x=[Tmin,Tmax]
    y=[Tmin,Tmax]
    y105=[1.05*Tmin,1.05*Tmax]
    y95=[0.95*Tmin,0.95*Tmax]
    
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)    
    plt.xlim(Tmin,Tmax)
    plt.ylim(Tmin,Tmax)
    plt.legend(loc=2,fontsize=9)
    plt.tight_layout(pad=0.2)        
    plt.tick_params(direction='in')
    plt.show()
    fig.savefig('ANN_m_suc.pdf')
    
    # Validation m_inj
    fig=pylab.figure(figsize=(4,4))

    plt.plot(m_inj_ANN[:n_training],m_inj[:n_training],'ro',ms = 3,mec='black',mew=0.5,label='Training points')
    plt.plot(m_inj_ANN[-n_split:],m_inj[-n_split:],'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
    plt.text(0.04,0.01,'R$^2$ = {:0.01f}%\n'.format(Rsquared(m_inj,m_inj_ANN)*100)+'MAE = {:0.01f}%\n'.format(mape(m_inj_ANN,m_inj))+'RMSE = {:0.01f}%\n'.format(rmse(m_inj_ANN,m_inj)),ha='left',va='center',fontsize = 8)

    plt.xlabel('$\dot m_{inj,pred}$ [kg/s]')
    plt.ylabel('$\dot m_{inj,exp}$ [kg/s]')

    Tmin = 0
    Tmax = 0.06
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
    fig.savefig('ANN_m_inj.pdf')
    
    # Validation m_tot
    fig=pylab.figure(figsize=(4,4))

    plt.plot(m_tot_ANN[:n_training],m_tot[:n_training],'ro',ms = 3,mec='black',mew=0.5,label='Training points')
    plt.plot(m_tot_ANN[-n_split:],m_tot[-n_split:],'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
    plt.text(0.125,0.05,'R$^2$ = {:0.01f}%\n'.format(Rsquared(m_tot,m_tot_ANN)*100)+'MAE = {:0.01f}%\n'.format(mape(m_tot_ANN,m_tot))+'RMSE = {:0.01f}%\n'.format(rmse(m_tot_ANN,m_tot)),ha='left',va='center',fontsize = 8)

    plt.xlabel('$\dot m_{tot,pred}$ [kg/s]')
    plt.ylabel('$\dot m_{tot,exp}$ [kg/s]')

    Tmin = 0
    Tmax = 0.2
    x=[Tmin,Tmax]
    y=[Tmin,Tmax]
    y105=[1.05*Tmin,1.05*Tmax]
    y95=[0.95*Tmin,0.95*Tmax]
    
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)    
    plt.xlim(Tmin,Tmax)
    plt.ylim(Tmin,Tmax)
    plt.legend(loc=2,fontsize=9)
    plt.tight_layout(pad=0.2)        
    plt.tick_params(direction='in')
    plt.show()
    fig.savefig('ANN_m_tot.pdf')
    
    # Validation W_comp
    fig=pylab.figure(figsize=(4,4))

    plt.plot(W_comp_ANN[:n_training],W_comp[:n_training],'ro',ms = 3,mec='black',mew=0.5,label='Training points')
    plt.plot(W_comp_ANN[-n_split:],W_comp[-n_split:],'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
    plt.text(6,3,'R$^2$ = {:0.01f}%\n'.format(Rsquared(W_comp,W_comp_ANN)*100)+'MAE = {:0.01f}%\n'.format(mape(W_comp_ANN,W_comp))+'RMSE = {:0.01f}%\n'.format(rmse(W_comp_ANN,W_comp)),ha='left',va='center',fontsize = 8)

    plt.xlabel('$\dot W_{comp,pred}$ [kW]')
    plt.ylabel('$\dot W_{comp,exp}$ [kW]')

    Tmin = 2
    Tmax = 8
    x=[Tmin,Tmax]
    y=[Tmin,Tmax]
    y105=[1.05*Tmin,1.05*Tmax]
    y95=[0.95*Tmin,0.95*Tmax]
    
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)    
    plt.xlim(Tmin,Tmax)
    plt.ylim(Tmin,Tmax)
    plt.legend(loc=2,fontsize=9)
    plt.tight_layout(pad=0.2)        
    plt.tick_params(direction='in')
    plt.show()
    fig.savefig('ANN_W_comp.pdf')
    
    # Validation f_loss
    fig=pylab.figure(figsize=(4,4))

    plt.plot(f_loss_ANN[:n_training],f_loss[:n_training],'ro',ms = 3,mec='black',mew=0.5,label='Training points')
    plt.plot(f_loss_ANN[-n_split:],f_loss[-n_split:],'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
    plt.text(6,3,'R$^2$ = {:0.01f}%\n'.format(Rsquared(f_loss,f_loss_ANN)*100)+'MAE = {:0.01f}%\n'.format(mape(f_loss_ANN,f_loss))+'RMSE = {:0.01f}%\n'.format(rmse(f_loss_ANN,f_loss)),ha='left',va='center',fontsize = 8)

    plt.xlabel('$f_{loss,pred}$ [%]')
    plt.ylabel('$f_{loss,exp}$ [%]')

    Tmin = 2
    Tmax = 8
    x=[Tmin,Tmax]
    y=[Tmin,Tmax]
    y105=[1.05*Tmin,1.05*Tmax]
    y95=[0.95*Tmin,0.95*Tmax]
    
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)    
    plt.xlim(Tmin,Tmax)
    plt.ylim(Tmin,Tmax)
    plt.legend(loc=2,fontsize=9)
    plt.tight_layout(pad=0.2)        
    plt.tick_params(direction='in')
    plt.show()
    fig.savefig('ANN_f_loss.pdf')
    
    print 'T_dis:',REmean(T_dis,T_dis_ANN),Rsquared(T_dis,T_dis_ANN)*100
    print 'm_suc:',REmean(m_suc,m_suc_ANN),Rsquared(m_suc,m_suc_ANN)*100
    print 'm_inj:',REmean(m_inj,m_inj_ANN),Rsquared(m_inj,m_inj_ANN)*100
    print 'm_tot:',REmean(m_tot,m_tot_ANN),Rsquared(m_tot,m_tot_ANN)*100
    print 'W_comp:',REmean(W_comp,W_comp_ANN),Rsquared(W_comp,W_comp_ANN)*100
    print 'f_loss:',REmean(f_loss,f_loss_ANN),Rsquared(f_loss,f_loss_ANN)*100

    
if __name__ == '__main__':
    
    Calculate()
    
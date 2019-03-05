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

from keras import backend as K

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
    T_db_in = float(data[i][0])
    T_wb_in = float(data[i][1])
    T_db_out = float(data[i][2])
    T_wb_out = float(data[i][3])
    T_sub = float(data[i][4])
    T_sup = float(data[i][5])
    T_cond = float(data[i][6])
    T_evap = float(data[i][7])
    Ref_type = str(data[i][8])
    Q  = float(data[i][9])
    W_tot = float(data[i][10])
    COP = float(data[i][11])
    
    i=i+1
    
    while i < (end - start+1):
        T_db_in = np.append(T_db_in,float(data[i][0]))
        T_wb_in = np.append(T_wb_in,float(data[i][1]))
        T_db_out = np.append(T_db_out,float(data[i][2])) 
        T_wb_out = np.append(T_wb_out,float(data[i][3]))
        T_sub = np.append(T_sub,float(data[i][4]))
        T_sup = np.append(T_sup,float(data[i][5]))
        T_cond = np.append(T_cond,float(data[i][6]))
        T_evap = np.append(T_evap,float(data[i][7]))
        Ref_type = np.append(Ref_type,str(data[i][8]))
        Q = np.append(Q,float(data[i][9]))
        W_tot = np.append(W_tot,float(data[i][10]))
        COP = np.append(COP,float(data[i][11]))
        i=i+1
        Data = [T_db_in,T_wb_in,T_db_out,T_wb_out,T_sub,T_sup,T_cond,T_evap,Ref_type,Q,W_tot,COP]
    
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

def coeff_determination(y_true, y_pred):
    
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    
def Calculate():

    "Import Experimental Data"
    start=1
    end=126
    filename = 'Data_Collection_modifed.csv'
    
    #Define inputs
    [T_db_in,T_wb_in,T_db_out,T_wb_out,T_sub,T_sup,T_cond,T_evap,Ref_type,Q_exp,W_tot_exp,COP_exp] = Import(start,end,filename)
    
    mode = 'training'
    
    from keras.models import Model,Sequential,load_model
    from keras.layers import Input,Flatten,Dense,LSTM,merge,Dropout,concatenate
    #from keras.engine import merge # from Keras version 1.2.2
    #from keras.layers.merge import concatenate
    from keras.layers import GaussianNoise
    from keras.utils import plot_model
    from keras.callbacks import TensorBoard
    from keras import regularizers
    
    
    #Normalize all parameters
    T_db_in_norm = Normalize(T_db_in, 0,26.7)
    T_wb_in_norm = Normalize(T_wb_in, 0,19.4)
    T_db_out_norm = Normalize(T_db_out, 35,48)
    T_wb_out_norm = Normalize(T_wb_out, 0,23.9)
    T_sub_norm = Normalize(T_sub, 0.29,16.95)
    T_sup_norm = Normalize(T_sup, 3.324,31.039)
    T_cond_norm = Normalize(T_cond, 44.56,65.98)
    T_evap_norm = Normalize(T_evap, -5.58,13.25)
    Q_exp_norm = Normalize(Q_exp, 4.66443073,36.58203557)
    W_tot_exp_norm = Normalize(W_tot_exp, 2.07,16.31)
    #COP_exp_norm = Normalize(COP_exp, 1.519597253,3.12397863)
    
    #convert to numpy array
    T_db_in_norm = np.array(T_db_in_norm)
    T_wb_in_norm = np.array(T_wb_in_norm)
    T_db_out_norm = np.array(T_db_out_norm)
    T_wb_out_norm = np.array(T_wb_out_norm)
    T_sub_norm = np.array(T_sub_norm)
    T_sup_norm = np.array(T_sup_norm)
    T_cond_norm = np.array(T_cond_norm)
    T_evap_norm = np.array(T_evap_norm)
    Q_exp_norm = np.array(Q_exp_norm)
    W_tot_exp_norm = np.array(W_tot_exp_norm)
    #COP_exp_norm = np.array(COP_exp_norm)
    
    # split into input (X) and output (Y) variables
    X = np.column_stack((T_db_in_norm, T_wb_in_norm))
    X = np.column_stack((X, T_db_out_norm))
    X = np.column_stack((X, T_wb_out_norm))
    X = np.column_stack((X, T_sub_norm))
    X = np.column_stack((X, T_sup_norm))
    #X = np.column_stack((T_sub_norm, T_sup_norm))
    X = np.column_stack((X, T_cond_norm))
    X = np.column_stack((X, T_evap_norm))
    #Y = np.column_stack((Q_exp_norm, COP_exp_norm))
    Y = np.column_stack((Q_exp_norm, W_tot_exp_norm))
    
    from sklearn.model_selection import train_test_split
    # shuffle the data before splitting for validation
    #X_remain, X_valid, Y_remain, Y_valid = train_test_split(X, Y, test_size=0.15, shuffle= True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle= True)
    
    SC = np.array([]) #empty score array
    
    for i in range(1):
        if mode == 'training':
            # create model
            model = Sequential()
            model.add(Dense(i+8, input_dim=8, activation='tanh')) #init='uniform' #use_bias = True, bias_initializer='zero' #4 is perfect
            #model.add(GaussianNoise(0.1))
            #model.add(Dropout(0.2)) #Dropout is a technique where randomly selected neurons are ignored during training.
            model.add(Dense(8, activation='tanh'))
            #model.add(GaussianNoise(0.1))
            model.add(Dense(8, activation='tanh'))
            model.add(Dense(8, activation='tanh'))
            model.add(Dense(8, activation='tanh'))
            model.add(Dense(8, activation='tanh'))
            model.add(Dense(2, activation='linear'))
              
            plot_model(model, to_file='model.pdf',show_shapes=True,show_layer_names=True)
      
            # Compile model
            model.compile(optimizer='adamax',loss='mse',metrics=['mae',coeff_determination])
              
            # fit the model
            history = model.fit(X_train,
                                Y_train,
                                epochs=4000 , #Cut the epochs in half when using sequential 
                                batch_size=20, #increase the batch size results in faster compiler an d high error, while smaller batch size results in slower compiler and slightly accurate model
                                #validation_split=0.2,
                                validation_data=(X_test,Y_test),
                                shuffle=True, #this is always set as True, even if not specified
                                )    
              
            
                
        #   #History plot for loss
            fig=pylab.figure(figsize=(6,4))
            plt.semilogy(history.history['loss'])
            plt.semilogy(history.history['val_loss'])
            plt.ylabel('MSE')
            plt.xlabel('epochs')
            plt.legend(['Train', 'Test'], loc='upper right',fontsize=9)
            #plt.ylim(0,0.1)
            plt.tight_layout(pad=0.2)  
            plt.tick_params(direction='in')      
            fig.savefig('ANN_history_GA_loss.pdf')
    
        #   #History plot for accuracy
            fig=pylab.figure(figsize=(6,4))
            plt.semilogy(history.history['coeff_determination'])
            plt.semilogy(history.history['val_coeff_determination'])
            plt.ylabel('R$^2$')
            plt.xlabel('epochs')
            plt.legend(['Train', 'Test'], loc='upper right',fontsize=9)
            #plt.ylim(0,0.1)
            plt.tight_layout(pad=0.2)  
            plt.tick_params(direction='in')      
            fig.savefig('ANN_history_GA_acc.pdf')
                    
            # Save the model
            model.save('ANN_model_GA.h5')
        
        elif mode == 'run':
        
            # Load the model
            model = load_model('ANN_model_GA.h5',custom_objects={'coeff_determination': coeff_determination})
        
        # Run the model
        predictions = model.predict(X)
        Q_ANN = DeNormalize(predictions[:,0].reshape(-1), 4.66443073,36.58203557)
        #COP_ANN = DeNormalize(predictions[:,1].reshape(-1), 1.519597253,3.12397863)
        W_tot_ANN = DeNormalize(predictions[:,1].reshape(-1), 2.07,16.31)
        
        # evaluate the model (for the last batch)
        scores = model.evaluate(X,Y)
        SC = np.append(SC,scores[1]*100)
        print('')
        print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
           
        # extract the weight and bias
        weights = model.layers[0].get_weights()[0]
        biases = model.layers[0].get_weights()[1]
        print('')
        print 'weights = ', weights
        print 'biases = ', biases
        # Save the architecture of a model, and not its weights or its training configuration
        # save as JSON
        # json_string = model.to_json()
        
        # save as YAML
        # yaml_string = model.to_yaml()
        
        # to SAVE into excel file
    print('')
    for i in range(len(SC)):
        print (SC[i])
        
#     for i in range(0,(end-start+1)):
#  
#  
#         data_calc = {'Q_ANN':[Q_ANN[i]],'COP_ANN':[COP_ANN[i]]} 
#              
#          
#         # Write to Excel
#         filename = os.path.dirname(__file__)+'/GA_output.xlsx'
#         xl = pd.read_excel(filename, sheet_name='ANN_Validation')
#  
#         df = pd.DataFrame(data=data_calc)
#  
#         df.reindex(columns=xl.columns)
#         df_final=xl.append(df,ignore_index=True)
#         df_final.tail()
#          
#         book = load_workbook(filename)
#         writer = pd.ExcelWriter(filename, engine='openpyxl',index=False)
#         writer.book = book
#         writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
#         df_final.to_excel(writer,index=False,sheet_name='ANN_Validation')
#          
#         # 
#         writer.save()

    
    #Separate testing from calibrating
    sep_val = 0.2
    n_len = len(Q_ANN)
    n_split = int(np.floor(sep_val*n_len))
    n_training = int(n_len-n_split-1)


    # Validation Q
    fig=pylab.figure(figsize=(4,4))

    plt.plot(Q_ANN[:n_training],Q_exp[:n_training],'ro',ms = 3,mec='black',mew=0.5,label='Training points')
    plt.plot(Q_ANN[-n_split:],Q_exp[-n_split:],'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
    plt.text(25,10,'R$^2$ = {:0.01f}%\n'.format(Rsquared(Q_exp,Q_ANN)*100)+'MAE = {:0.01f}%\n'.format(mape(Q_ANN,Q_exp))+'RMSE = {:0.01f}%\n'.format(rmse(Q_ANN,Q_exp)),ha='left',va='center',fontsize = 8)

    plt.xlabel('$\dot Q_{pred}$ [kW]')
    plt.ylabel('$\dot Q_{exp}$ [kW]')

    Tmin = 0
    Tmax = 40
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
    fig.savefig('ANN_Q.pdf')
    
#     # Validation COP
#     fig=pylab.figure(figsize=(4,4))
# 
#     plt.plot(COP_ANN[:n_training],COP_exp[:n_training],'ro',ms = 3,mec='black',mew=0.5,label='Training points')
#     plt.plot(COP_ANN[-n_split:],COP_exp[-n_split:],'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
#     plt.text(3,1.5,'R$^2$ = {:0.01f}%\n'.format(Rsquared(COP_exp,COP_ANN)*100)+'MAE = {:0.01f}%\n'.format(mape(COP_ANN,COP_exp))+'RMSE = {:0.01f}%\n'.format(rmse(COP_ANN,COP_exp)),ha='left',va='center',fontsize = 8)
# 
#     plt.xlabel('COP [-]')
#     plt.ylabel('COP [-]')
# 
#     Tmin = 1
#     Tmax = 4
#     x=[Tmin,Tmax]
#     y=[Tmin,Tmax]
#     y105=[1.1*Tmin,1.1*Tmax]
#     y95=[0.9*Tmin,0.9*Tmax]
#     
#     plt.plot(x,y,'k-')
#     plt.fill_between(x,y105,y95,color='black',alpha=0.2)    
#     plt.xlim(Tmin,Tmax)
#     plt.ylim(Tmin,Tmax)
#     plt.legend(loc=2,fontsize=9)
#     plt.tight_layout(pad=0.2)        
#     plt.tick_params(direction='in')
#     plt.show()
#     fig.savefig('ANN_COP.pdf')
    
    # Validation W
    fig=pylab.figure(figsize=(4,4))

    plt.plot(W_tot_ANN[:n_training],W_tot_exp[:n_training],'ro',ms = 3,mec='black',mew=0.5,label='Training points')
    plt.plot(W_tot_ANN[-n_split:],W_tot_exp[-n_split:],'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
    plt.text(12,4,'R$^2$ = {:0.01f}%\n'.format(Rsquared(W_tot_exp,W_tot_ANN)*100)+'MAE = {:0.01f}%\n'.format(mape(W_tot_ANN,W_tot_exp))+'RMSE = {:0.01f}%\n'.format(rmse(W_tot_ANN,W_tot_exp)),ha='left',va='center',fontsize = 8)

    plt.xlabel('$\dot W_{pred}$ [kW]')
    plt.ylabel('$\dot W_{exp}$ [kW]')

    Tmin = 1
    Tmax = 18
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
    fig.savefig('ANN_W_tot.pdf')
    
    print 'Q:',REmean(Q_exp,Q_ANN),Rsquared(Q_exp,Q_ANN)*100 
#     print 'COP:',REmean(COP_exp,COP_ANN),Rsquared(COP_exp,COP_ANN)*100 
    print 'W_tot:',REmean(W_tot_exp,W_tot_ANN),Rsquared(W_tot_exp,W_tot_ANN)*100 


    #Validation with Shikha's data
#     "Import Experimental Data"
#     start=1
#     end=48
#     filename = 'Data_shikha.csv'
#     #Define inputs
#     [Tmin_exp_shikha,Tsub,Psat,Mat,LD,Bf,Bw,BfBw] = Import(start,end,filename)
#     #Normalize all parameters
#     Tmin_exp_norm = Normalize(Tmin_exp_shikha, 206.8841, 727.8873239)
#     Tsub_norm = Normalize(Tsub, 0, 39.84150546)
#     Psat_norm = Normalize(Psat, 0.001185867, 3.003378378)
#     LD_norm = Normalize(LD, 2.67, 63.5)
#     Bf_norm = Normalize(Bf, 2428162.849, 2744290.164)
#     Bw_norm = Normalize(Bw, 5168800, 1379121205)
#     BfBw_norm = Normalize(BfBw, 0.001989845, 0.530923555)
#     #convert to numpy array
#     Tmin_exp_norm = np.array(Tmin_exp_norm)
#     Tsub_norm = np.array(Tsub_norm)
#     Psat_norm = np.array(Psat_norm)
#     LD_norm = np.array(LD_norm)
#     Bf_norm = np.array(Bf_norm)
#     Bw_norm = np.array(Bw_norm)
#     BfBw_norm = np.array(BfBw_norm)
#     # split into input (X) and output (Y) variables
#     X = np.column_stack((Tsub_norm, Psat_norm))
#     X = np.column_stack((X, LD_norm))
#     X = np.column_stack((X, BfBw_norm))
#     # Load the model
#     model = load_model('ANN_model_Tmin.h5')
#     # Run the model
#     Tmin_ANN_shikha = model.predict(X_valid)
#     Tmin_ANN_shikha = DeNormalize(Tmin_ANN_shikha.reshape(-1), 206.8841, 727.8873239)
#     
#     Tmin_exp_valid = DeNormalize(Y_valid.reshape(-1), 206.8841, 727.8873239) 
#     
#     # New Validation Tmin of shikha
#     fig=pylab.figure(figsize=(4,4))
#  
#     plt.plot(Tmin_ANN[:n_training],Tmin_exp[:n_training],'ro',ms = 3,mec='black',mew=0.5,label='Training points')
#     plt.plot(Tmin_ANN[-n_split:],Tmin_exp[-n_split:],'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
#     plt.plot(Tmin_ANN_shikha,Tmin_exp_valid,'g^',ms = 4,mec='black',mew=0.5,label='Validation points')
#     plt.text(550,200,'R$^2$ = {:0.01f}%\n'.format(Rsquared(Tmin_exp_valid,Tmin_ANN_shikha)*100)+'MAE = {:0.01f}%\n'.format(mape(Tmin_ANN_shikha,Tmin_exp_valid))+'RMSE = {:0.01f}%\n'.format(rmse(Tmin_ANN_shikha,Tmin_exp_valid)),ha='left',va='center',fontsize = 8)
#  
#     plt.xlabel('$T_{min,pred}$ [$\degree$C]')
#     plt.ylabel('$T_{min,exp}$ [$\degree$C]')
#  
#     Tmin = 100
#     Tmax = 800
#     x=[Tmin,Tmax]
#     y=[Tmin,Tmax]
#     y105=[1.1*Tmin,1.1*Tmax]
#     y95=[0.9*Tmin,0.9*Tmax]
#      
#     plt.plot(x,y,'k-')
#     plt.fill_between(x,y105,y95,color='black',alpha=0.2)    
#     plt.xlim(Tmin,Tmax)
#     plt.ylim(Tmin,Tmax)
#     plt.legend(loc=2,fontsize=9)
#     plt.tight_layout(pad=0.2)        
#     plt.tick_params(direction='in')
#     plt.show()
#     fig.savefig('ANN_Tmin_shikha.pdf')
    
    
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
    
    
    
    
    
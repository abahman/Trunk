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

    [T_db_in,T_wb_in,T_db_out,T_wb_out,T_sub,T_sup,T_cond,T_evap,Ref_type,Q_exp,W_tot_exp,COP_exp] = Import(start,end,filename)
    

    mode = 'training'
    
    from keras.models import Model,Sequential,load_model
    from keras.layers import Input,Flatten,Dense,LSTM,merge,Dropout,concatenate
    #from keras.engine import merge # from Keras version 1.2.2
    #from keras.layers.merge import concatenate
    from keras.utils import plot_model
    from keras.callbacks import TensorBoard
    from keras import regularizers
    

    #Define inputs
    #T_db_in = T_db_in.reshape(-1,1,1)
    #T_wb_in = T_wb_in.reshape(-1,1,1)
    #T_db_out = T_db_out.reshape(-1,1,1)
    #T_wb_out = T_wb_out.reshape(-1,1,1)
    T_sub = T_sub.reshape(-1,1,1)
    T_sup = T_sup.reshape(-1,1,1)
    T_cond = T_cond.reshape(-1,1,1)
    T_evap = T_evap.reshape(-1,1,1)
    
    
    #Normalize all parameters
    #T_db_in_norm = Normalize(T_db_in, 26.7,26.7)
    #T_wb_in_norm = Normalize(T_wb_in, 19.4,19.4)
    #T_db_out_norm = Normalize(T_db_out, 35,48)
    #T_wb_out_norm = Normalize(T_wb_out, 23.9,23.9)
    T_sub_norm = Normalize(T_sub, 0.29,16.95)
    T_sup_norm = Normalize(T_sup, 3.324,31.039)
    T_cond_norm = Normalize(T_cond, 44.56,65.98)
    T_evap_norm = Normalize(T_evap, -5.58,13.25)
    Q_exp_norm = Normalize(Q_exp, 4.66443073,36.58203557)
    #W_tot_exp_norm = Normalize(W_tot_exp, 2.07,16.31)
    COP_exp_norm = Normalize(COP_exp, 1.519597253,3.12397863)
    
    if mode == 'training':
        #visible1 = Input(shape=(1,1), name='T_db_in')
        #visible2 = Input(shape=(1,1), name='T_wb_in')
        #visible3 = Input(shape=(1,1), name='T_db_out')
        #visible4 = Input(shape=(1,1), name='T_wb_out')
        visible5 = Input(shape=(1,1), name='T_sub')
        visible6 = Input(shape=(1,1), name='T_sup')
        visible7 = Input(shape=(1,1), name='T_cond')
        visible8 = Input(shape=(1,1), name='T_evap')
    
        shared_lstm = LSTM(4)
    
        #encoded_a = shared_lstm(visible1)
        #encoded_b = shared_lstm(visible2)
        #encoded_c = shared_lstm(visible3)
        #encoded_d = shared_lstm(visible4)
        encoded_e = shared_lstm(visible5)
        encoded_f = shared_lstm(visible6)
        encoded_g = shared_lstm(visible7)
        encoded_h = shared_lstm(visible8)

    
        #Merge inputs
        merged = concatenate([encoded_e,encoded_f,encoded_g,encoded_h],axis=-1)
        
        #interpretation model
        hidden1 = Dense(10,activation='tanh')(merged) #hidden1 = Dense(256, activation='tanh')(merged) ###'relu' shows good results
        #hidden2 = Dense(10, activation = 'tanh')(hidden1)
        #hidden3 = Dropout(0.2, noise_shape=None, seed=None)(hidden2)
        #hidden3 = Dense(100, activation = 'tanh')(hidden2)
        #hidden4 = Dense(32, activation = 'tanh')(hidden3)
        output1 = Dense(1, activation = 'linear',name='Q')(hidden1)
        output2 = Dense(1, activation = 'linear',name='COP')(hidden1)

       
        model = Model(input=[visible5,visible6,visible7,visible8],
                        output = [output1,output2])
        
        plot_model(model, to_file='model.pdf',show_shapes=True,show_layer_names=True)
        
        model.compile(optimizer='adamax',loss=['mse','mse'],metrics=['mae',coeff_determination]) #model.compile(optimizer='adamax',loss=['mse','mse','mse','mse','mse','mse']) #metrics are not included in the training
        
        X = [T_sub_norm,T_sup_norm,T_cond_norm,T_evap_norm]
        Y = [Q_exp_norm,COP_exp_norm]
        history = model.fit(X,
                            Y,
                            epochs=2000 ,
                            batch_size=15, #increase the batch size results in faster compiler an d high error, while smaller batch size results in slower compiler and slightly accurate model
                            validation_split=0.2,
                            )
        

        
    #   #History plot
        fig=pylab.figure(figsize=(6,4))
        plt.semilogy(history.history['loss'])
        plt.semilogy(history.history['val_loss'])
        #plt.semilogy(history.history['mae'])
        plt.ylabel('loss [-]')
        plt.xlabel('epoch [-]')
        plt.legend(['Train', 'Test'], loc='upper right',fontsize=9)
        #plt.ylim(0,0.1)
        plt.tight_layout(pad=0.2)  
        plt.tick_params(direction='in')      
        fig.savefig('ANN_history_GA.pdf')
        
        # Save the model
        model.save('ANN_model_GA.h5')
    
    elif mode == 'run':
    
        # Load the model
        model = load_model('ANN_model_GA.h5')
    
    # Run the model
    [Q_ANN,COP_ANN] = model.predict(X)
    Q_ANN = DeNormalize(Q_ANN.reshape(-1), 4.66443073,36.58203557)
    COP_ANN = DeNormalize(COP_ANN.reshape(-1), 1.519597253,3.12397863)
    
    # evaluate the model
    scores = model.evaluate(X,Y)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    
    # extract the weight and bias
#     weights = model.layers[0].get_weights()[0]
#     biases = model.layers[0].get_weights()[1]
#     print('')
#     print 'weights = ', weights
#     print 'biases = ', biases    

    # Save the architecture of a model, and not its weights or its training configuration
    # save as JSON
    # json_string = model.to_json()
    
    # save as YAML
    # yaml_string = model.to_yaml()

    # to SAVE into excel file
    for i in range(0,(end-start+1)):
 
 
        data_calc = {'Q_ANN':[Q_ANN[i]],'COP_ANN':[COP_ANN[i]]} 
             
         
        # Write to Excel
        filename = os.path.dirname(__file__)+'/GA_output.xlsx'
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
    n_len = len(Q_ANN)
    n_split = int(np.floor(sep_val*n_len))
    n_training = int(n_len-n_split-1)


    # Validation Q
    fig=pylab.figure(figsize=(4,4))

    plt.plot(Q_ANN[:n_training],Q_exp[:n_training],'ro',ms = 3,mec='black',mew=0.5,label='Training points')
    plt.plot(Q_ANN[-n_split:],Q_exp[-n_split:],'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
    plt.text(25,10,'R$^2$ = {:0.01f}%\n'.format(Rsquared(Q_exp,Q_ANN)*100)+'MAE = {:0.01f}%\n'.format(mape(Q_ANN,Q_exp))+'RMSE = {:0.01f}%\n'.format(rmse(Q_ANN,Q_exp)),ha='left',va='center',fontsize = 8)

    plt.xlabel('$Q_{pred}$ [kW]')
    plt.ylabel('$Q_{exp}$ [kW]')

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
    
    # Validation COP
    fig=pylab.figure(figsize=(4,4))

    plt.plot(COP_ANN[:n_training],COP_exp[:n_training],'ro',ms = 3,mec='black',mew=0.5,label='Training points')
    plt.plot(COP_ANN[-n_split:],COP_exp[-n_split:],'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
    plt.text(3,1.5,'R$^2$ = {:0.01f}%\n'.format(Rsquared(COP_exp,COP_ANN)*100)+'MAE = {:0.01f}%\n'.format(mape(COP_ANN,COP_exp))+'RMSE = {:0.01f}%\n'.format(rmse(COP_ANN,COP_exp)),ha='left',va='center',fontsize = 8)

    plt.xlabel('COP [-]')
    plt.ylabel('COP [-]')

    Tmin = 1
    Tmax = 4
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
    fig.savefig('ANN_COP.pdf')
    
    print 'Q:',REmean(Q_exp,Q_ANN),Rsquared(Q_exp,Q_ANN)*100 
    print 'COP:',REmean(COP_exp,COP_ANN),Rsquared(COP_exp,COP_ANN)*100 


    
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
    
    
    
    
    
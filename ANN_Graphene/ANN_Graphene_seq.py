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

import pickle

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
    visc_exp = float(data[i][0])
    Temp = float(data[i][1])
    graphene_vol = float(data[i][2])
    surfactant_rho = float(data[i][3])
    surfactant_vol = float(data[i][4])

    i=i+1
    
    while i < (end - start+1):
        visc_exp = np.append(visc_exp,float(data[i][0]))
        Temp = np.append(Temp,float(data[i][1]))
        graphene_vol = np.append(graphene_vol,float(data[i][2])) 
        surfactant_rho = np.append(surfactant_rho,float(data[i][3]))
        surfactant_vol = np.append(surfactant_vol,float(data[i][4]))

        i=i+1
        Data = [visc_exp,Temp,graphene_vol,surfactant_rho,surfactant_vol]  

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
    return round(MAPE,4)

def mse(y_pred, y_true):
    '''
    Mean Squared Error
    '''
    from sklearn.metrics import mean_squared_error
    MSE = mean_squared_error(y_pred, y_true)
    return round(MSE,4)

def Normalize(y_data,y_data_min,y_data_max):
    
    y_norm = 0.8*(y_data - y_data_min)/(y_data_max - y_data_min) + 0.1
    
    return y_norm

def DeNormalize(y_norm,y_data_min,y_data_max):
    
    y = (y_norm - 0.1)*(y_data_max - y_data_min)/0.8 + y_data_min
    
    return y

def REmean(y_true,y_pred):
    "same as Mean absolute error"
    
    return np.mean(np.fabs(y_true - y_pred)/y_true)    

def Rsquared(y_true,y_pred):
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true,y_pred)    
    
    return round(r_value**2,4)

def coeff_determination(y_true, y_pred):
    
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    
def Calculate():

    "Import Experimental Data"
    start=1
    end=88
    filename = 'Data_Collection_new.csv'
    
    #Define inputs
    #[Tmin_exp,Tsub,Psat,Mat,LD,Bf,Bw,BfBw,Tmin_Mori,Tmin_Adler,Tmin_Dhir,Tmin_Lauer,Tmin_Freud,Tmin_shikha_7,Tmin_shikha_8,Tmin_shikha_9,Tmin_Brenson,Tmin_Henry,Tmin_Perenson,Tmin_Shikha,Tmin_Sakurai,Tmin_Henry_1atm,Tmin_exp_1atm,Tmin_exp2_1atm,Tmin_ANN_1atm] = Import(start,end,filename)
    [visc_exp,Temp,graphene_vol,surfactant_rho,surfactant_vol] = Import(start,end,filename)
    
    mode = 'training'
    
    from keras.models import Model,Sequential,load_model
    from keras.layers import Input,Flatten,Dense,LSTM,merge,Dropout,concatenate
    #from keras.engine import merge # from Keras version 1.2.2
    #from keras.layers.merge import concatenate
    from keras.layers import GaussianNoise
    from keras.utils.vis_utils import plot_model
    from keras.callbacks import TensorBoard
    from keras import regularizers
    
    
    #Normalize all parameters
    visc_exp_norm = Normalize(visc_exp, 0.452954654, 1.269923504)
    Temp_norm = Normalize(Temp, 20, 50)
    graphene_vol_norm = Normalize(graphene_vol, 0.0, 0.05303025)
    surfactant_rho_norm = Normalize(surfactant_rho, 0.0, 1.4)
    surfactant_vol_norm = Normalize(surfactant_vol, 0.0, 0.079545375)
    
    #convert to numpy array
    visc_exp_norm = np.array(visc_exp_norm)
    Temp_norm = np.array(Temp_norm)
    graphene_vol_norm = np.array(graphene_vol_norm)
    surfactant_rho_norm = np.array(surfactant_rho_norm)
    surfactant_vol_norm = np.array(surfactant_vol_norm)
    
    # split into input (X) and output (Y) variables
    X = np.column_stack((Temp_norm, graphene_vol_norm))
    X = np.column_stack((X, surfactant_rho_norm))
    X = np.column_stack((X, surfactant_vol_norm))
    Y = visc_exp_norm
    
    from sklearn.model_selection import train_test_split
    # shuffle the data before splitting for validation
    X_remain, X_valid, Y_remain, Y_valid = train_test_split(X, Y, test_size=0.15, shuffle= True)
    X_train, X_test, Y_train, Y_test = train_test_split(X_remain, Y_remain, test_size=0.15, shuffle= True)
    
    SC = np.array([]) #empty score array
    ms = np.array([])
    ma = np.array([])
    R2 = np.array([])
    
    for i in range(1):
        if mode == 'training':
            # create model
            model = Sequential()
            model.add(Dense(i+9, input_dim=4, activation='tanh')) #init='uniform' #use_bias = True, bias_initializer='zero' #4 is perfect
            #model.add(GaussianNoise(0.1))
            #model.add(Dropout(0.2)) #Dropout is a technique where randomly selected neurons are ignored during training.
#             model.add(Dense(i+12, activation='tanh'))
            #model.add(GaussianNoise(0.1))
            #model.add(Dense(i+12, activation='tanh'))
            model.add(Dense(1, activation='linear'))
              
            #plot_model(model, to_file='model.pdf',show_shapes=True,show_layer_names=True)
      
            # Compile model
            model.compile(optimizer='adamax',loss='mse',metrics=['mae',coeff_determination])
              
            # fit the model
            history = model.fit(X_train,
                                Y_train,
                                epochs=4000 , #Cut the epochs in half when using sequential 
                                batch_size=10, #increase the batch size results in faster compiler and high error, while smaller batch size results in slower compiler and slightly accurate model
                                #validation_split=0.2,
                                validation_data=(X_test,Y_test),
                                shuffle=True, #this is always set as True, even if not specified
                                #use_multiprocessing=False,
                                )    
                
        #   #History plot for loss
            fig=pylab.figure(figsize=(6,4))
            plt.semilogy(history.history['loss'])
            plt.semilogy(history.history['val_loss'])
            plt.ylabel('MSE')
            plt.xlabel('epochs')
            plt.legend(['Train', 'Test'], loc='upper right',fontsize=12)
            plt.ylim(0.001,1)
            plt.tight_layout(pad=0.2)  
            plt.tick_params(direction='in') 
#             plt.show()     
            fig.savefig('ANN_history_visc_loss.pdf')
    
        #   #History plot for accuracy
            fig=pylab.figure(figsize=(6,4))
            plt.semilogy(history.history['coeff_determination'])
            plt.semilogy(history.history['val_coeff_determination'])
            plt.ylabel('R$^2$')
            plt.xlabel('epochs')
            plt.legend(['Train', 'Test'], loc='upper right',fontsize=12)
            plt.ylim(0.01,1)
            plt.tight_layout(pad=0.2)  
            plt.tick_params(direction='in')
#             plt.show()     
            fig.savefig('ANN_history_visc_acc.pdf')
                    
            # Save the model
            model.save('ANN_model_visc.h5')
            # Save the history
            H = history.history
            with open('HistoryDict.pickle', 'wb') as handle:
                pickle.dump(H, handle)
                
        elif mode == 'run':
        
            # Load the model
            model = load_model('ANN_model_visc.h5',custom_objects={'coeff_determination': coeff_determination})
            # Load the history
            with open('HistoryDict.pickle', 'rb') as handle:
                H = pickle.load(handle)
  
        # Run the model
        visc_ANN = model.predict(X)
        visc_ANN = DeNormalize(visc_ANN.reshape(-1), 0.452954654, 1.269923504)
        
        # evaluate the model (for the last batch)
        scores = model.evaluate(X,Y)
        SC = np.append(SC,scores[1]*100)
        ms = np.append(ms,mse(visc_ANN,visc_exp))
        ma = np.append(ma,mape(visc_ANN, visc_exp))
        R2 = np.append(R2,Rsquared(visc_exp,visc_ANN))
        
        print('')
        print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
           
        # extract the weight and bias
        weights = model.layers[0].get_weights()[0]
        biases = model.layers[0].get_weights()[1]
        
        #to change the percision of printed numbers
        np.set_printoptions(precision=4, suppress=True,
                       threshold=10000,
                       linewidth=150)
        print('')
        print ('weights = ', weights.transpose())
        print ('biases = ', biases)

        # Save the architecture of a model, and not its weights or its training configuration
        # save as JSON
        # json_string = model.to_json()
        
        # save as YAML
        # yaml_string = model.to_yaml()
    print('')
    for i in range(len(SC)):
        print (SC[i])
        print (mape(visc_ANN[i], visc_exp[i]))
        print (ms[i],ma[i],R2[i])    
        
    # to SAVE into excel file
    for i in range(0,(end-start+1)):
  
        data_calc = {'visc':[visc_ANN[i]]}  #data_calc = {'Tdis':[T[i]],'mdot':[Mref[i]],'mdot_inj':[Minj[i]], 'Wdot':[W[i]],'etaoa':[eta_s[i]],'fq':[Q[i]/W[i]]} 
              
          
        # Write to Excel
        filename = os.path.dirname(__file__)+'/visc_output2.xlsx'
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
    sep_val = 0.15
    n_len = len(visc_ANN)
    n_split = int(np.floor(sep_val*n_len))
    n_training = int(n_len-n_split-1)


#     # Validation Tmin
#     fig=pylab.figure(figsize=(4,4))
# 
#     plt.plot(Tmin_ANN[:n_training],Tmin_exp[:n_training],'ro',ms = 3,mec='black',mew=0.5,label='Training points')
#     plt.plot(Tmin_ANN[-n_split:],Tmin_exp[-n_split:],'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
#     plt.text(550,200,'R$^2$ = {:0.01f}%\n'.format(Rsquared(Tmin_exp,Tmin_ANN)*100)+'MAE = {:0.01f}%\n'.format(mape(Tmin_ANN,Tmin_exp))+'RMSE = {:0.01f}%\n'.format(rmse(Tmin_ANN,Tmin_exp)),ha='left',va='center',fontsize = 8)
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
#     fig.savefig('ANN_Tmin.pdf')

    print ('Method:','MSE','MAPE','Rsquared')
    print ('visc_ANN:',mse(visc_ANN,visc_exp),mape(visc_ANN, visc_exp),Rsquared(visc_exp,visc_ANN))
#     print ('Tmin_Mori:',mse(Tmin_Mori,Tmin_exp),mape(Tmin_Mori, Tmin_exp),Rsquared(Tmin_exp,Tmin_Mori))
#     print ('Tmin_Adler:',mse(Tmin_Adler,Tmin_exp),mape(Tmin_Adler, Tmin_exp),Rsquared(Tmin_exp,Tmin_Adler))
#     print ('Tmin_Dhir:',mse(Tmin_Dhir,Tmin_exp),mape(Tmin_Dhir, Tmin_exp),Rsquared(Tmin_exp,Tmin_Dhir))
#     print ('Tmin_Lauer:',mse(Tmin_Lauer,Tmin_exp),mape(Tmin_Lauer, Tmin_exp),Rsquared(Tmin_exp,Tmin_Lauer))
#     print ('Tmin_Freud:',mse(Tmin_Freud,Tmin_exp),mape(Tmin_Freud, Tmin_exp),Rsquared(Tmin_exp,Tmin_Freud))
#     print ('Tmin_shikha_7:',mse(Tmin_shikha_7,Tmin_exp),mape(Tmin_shikha_7, Tmin_exp),Rsquared(Tmin_exp,Tmin_shikha_7))
#     print ('Tmin_shikha_8:',mse(Tmin_shikha_8,Tmin_exp),mape(Tmin_shikha_8, Tmin_exp),Rsquared(Tmin_exp,Tmin_shikha_8))
#     print ('Tmin_shikha_9:',mse(Tmin_shikha_9,Tmin_exp),mape(Tmin_shikha_9, Tmin_exp),Rsquared(Tmin_exp,Tmin_shikha_9))
#     print ('NEW')
#     print ('Tmin_Brenson:',mse(Tmin_Brenson,Tmin_exp),mape(Tmin_Brenson, Tmin_exp),Rsquared(Tmin_exp,Tmin_Brenson))
#     print ('Tmin_Henry:',mse(Tmin_Henry,Tmin_exp),mape(Tmin_Henry, Tmin_exp),Rsquared(Tmin_exp,Tmin_Henry))
#     print ('Tmin_Perenson:',mse(Tmin_Perenson,Tmin_exp),mape(Tmin_Perenson, Tmin_exp),Rsquared(Tmin_exp,Tmin_Perenson),rmse(Tmin_Perenson, Tmin_exp))
#     print ('Tmin_Shikha:',mse(Tmin_Shikha,Tmin_exp),mape(Tmin_Shikha, Tmin_exp),Rsquared(Tmin_exp,Tmin_Shikha),rmse(Tmin_Shikha, Tmin_exp))
#     print ('Tmin_Sakurai:',mse(Tmin_Sakurai,Tmin_exp),mape(Tmin_Sakurai, Tmin_exp),Rsquared(Tmin_exp,Tmin_Sakurai))
#     print ('Tmin_Henry_1atm:',mse(Tmin_Henry_1atm[0:119],Tmin_exp_1atm[0:119]),mape(Tmin_Henry_1atm[0:119], Tmin_exp_1atm[0:119]),Rsquared(Tmin_exp_1atm[0:119],Tmin_Henry_1atm[0:119]))
#     print ('Tmin_ANN_1atm:',mse(Tmin_ANN_1atm[0:152],Tmin_exp2_1atm[0:152]),mape(Tmin_ANN_1atm[0:152], Tmin_exp2_1atm[0:152]),Rsquared(Tmin_exp2_1atm[0:152],Tmin_ANN_1atm[0:152]))
    
#     for i in range(len(Tmin_ANN)): # print the measure absolute error (%) for all data
#         print (REmean(Tmin_exp[i],Tmin_ANN[i])*100)
    error = np.array([0,1,2,3,4,5,6,7,8,9,10,12,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
    set = np.array([0,14.51187335,27.70448549,41.16094987,50.92348285,61.7414248,72.55936675,81.26649077,86.80738786,89.97361478,93.13984169,95.77836412,97.88918206,99.73614776,99.73614776,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100])

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
    
    # Run the model
    #Validation
    visc_ANN_valid = model.predict(X_valid)
    visc_ANN_valid = DeNormalize(visc_ANN_valid.reshape(-1),  0.452954654, 1.269923504)
    visc_exp_valid = DeNormalize(Y_valid.reshape(-1),  0.452954654, 1.269923504) 
    #Training
    visc_ANN_train = model.predict(X_train)
    visc_ANN_train = DeNormalize(visc_ANN_train.reshape(-1), 0.452954654, 1.269923504)
    visc_exp_train = DeNormalize(Y_train.reshape(-1), 0.452954654, 1.269923504) 
    #Testing
    visc_ANN_test = model.predict(X_test)
    visc_ANN_test = DeNormalize(visc_ANN_test.reshape(-1), 0.452954654, 1.269923504)
    visc_exp_test = DeNormalize(Y_test.reshape(-1), 0.452954654, 1.269923504) 
        
    # plot all data
    fig=pylab.figure(figsize=(4,4))
    plt.plot(visc_ANN_train,visc_exp_train,'ro',ms = 4,mec='black',mew=0.5,label='Training points')
    plt.plot(visc_ANN_test,visc_exp_test,'b*',ms = 6,mec='black',mew=0.5,label='Testing points')
    plt.plot(visc_ANN_valid,visc_exp_valid,'g^',ms = 6,mec='black',mew=0.5,label='Validation points')
    plt.text(1,0.5,'R$^2$ = {:0.03f}\n'.format(Rsquared(visc_exp,visc_ANN))+'MAE = {:0.02f}%\n'.format(mape(visc_ANN,visc_exp))+'MSE = {:0.03f}\n'.format(mse(visc_ANN,visc_exp)),ha='left',va='center',fontsize = 10)
    plt.xlabel('$\\nu_{pred}$ [$-$]')
    plt.ylabel('$\\nu_{exp}$ [$-$]')
    Vmin = 0
    Vmax = 2
    x=[Vmin,Vmax]
    y=[Vmin,Vmax]
    y105=[1.1*Vmin,1.1*Vmax]
    y95=[0.9*Vmin,0.9*Vmax]
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)    
    plt.xlim(Vmin,Vmax)
    plt.ylim(Vmin,Vmax)
    plt.legend(loc=2,fontsize=12)
    plt.tight_layout(pad=0.2)        
    plt.tick_params(direction='in')
    plt.show()
    fig.savefig('ANN_visc_all.pdf')
    plt.close()
    
    # plots (training)
    fig=pylab.figure(figsize=(4,4))
    plt.plot(visc_ANN_train,visc_exp_train,'ro',ms = 4,mec='black',mew=0.5,label='Training points')
    plt.text(1,0.5,'R$^2$ = {:0.03f}\n'.format(Rsquared(visc_exp_train,visc_ANN_train))+'MAE = {:0.02f}%\n'.format(mape(visc_ANN_train,visc_exp_train))+'MSE = {:0.03f}\n'.format(mse(visc_ANN_train,visc_exp_train)),ha='left',va='center',fontsize = 10)
    plt.xlabel('$\\nu_{pred}$ [$-$]')
    plt.ylabel('$\\nu_{exp}$ [$-$]')
    Tmin = 0
    Tmax = 2
    x=[Tmin,Tmax]
    y=[Tmin,Tmax]
    y105=[1.1*Tmin,1.1*Tmax]
    y95=[0.9*Tmin,0.9*Tmax]
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)    
    plt.xlim(Tmin,Tmax)
    plt.ylim(Tmin,Tmax)
    plt.legend(loc=2,fontsize=12)
    plt.tight_layout(pad=0.2)        
    plt.tick_params(direction='in')
    plt.show()
    fig.savefig('ANN_visc_training.pdf')  
    plt.close()
     
    #  plots (testing)
    fig=pylab.figure(figsize=(4,4))
    plt.plot(visc_ANN_test,visc_exp_test,'b*',ms = 6,mec='black',mew=0.5,label='Testing points')
    plt.text(1,0.5,'R$^2$ = {:0.03f}\n'.format(Rsquared(visc_exp_test,visc_ANN_test))+'MAE = {:0.02f}%\n'.format(mape(visc_ANN_test,visc_exp_test))+'MSE = {:0.03f}\n'.format(mse(visc_ANN_test,visc_exp_test)),ha='left',va='center',fontsize = 10)
    plt.xlabel('$\\nu_{pred}$ [$-$]')
    plt.ylabel('$\\nu_{exp}$ [$-$]')
    Tmin = 0
    Tmax = 2
    x=[Tmin,Tmax]
    y=[Tmin,Tmax]
    y105=[1.1*Tmin,1.1*Tmax]
    y95=[0.9*Tmin,0.9*Tmax]
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)    
    plt.xlim(Tmin,Tmax)
    plt.ylim(Tmin,Tmax)
    plt.legend(loc=2,fontsize=12)
    plt.tight_layout(pad=0.2)        
    plt.tick_params(direction='in')
    plt.show()
    fig.savefig('ANN_visc_testing.pdf')
    plt.close()
     
    # plots (validation)
    fig=pylab.figure(figsize=(4,4))
    plt.plot(visc_ANN_valid,visc_exp_valid,'g^',ms = 6,mec='black',mew=0.5,label='Validation points')
    plt.text(1,0.5,'R$^2$ = {:0.03f}\n'.format(Rsquared(visc_exp_valid,visc_ANN_valid))+'MAE = {:0.02f}%\n'.format(mape(visc_ANN_valid,visc_exp_valid))+'MSE = {:0.03f}\n'.format(mse(visc_ANN_valid,visc_exp_valid)),ha='left',va='center',fontsize = 10)
    plt.xlabel('$\\nu_{pred}$ [$-$]')
    plt.ylabel('$\\nu_{exp}$ [$-$]')
    Tmin = 0
    Tmax = 2
    x=[Tmin,Tmax]
    y=[Tmin,Tmax]
    y105=[1.1*Tmin,1.1*Tmax]
    y95=[0.9*Tmin,0.9*Tmax]
    plt.plot(x,y,'k-')
    plt.fill_between(x,y105,y95,color='black',alpha=0.2)    
    plt.xlim(Tmin,Tmax)
    plt.ylim(Tmin,Tmax)
    plt.legend(loc=2,fontsize=12)
    plt.tight_layout(pad=0.2)        
    plt.tick_params(direction='in')
    plt.show()
    fig.savefig('ANN_visc_validation.pdf') 
    plt.close()
#     
#     # Cumulative Distribution of all data versus maximum absolute error
#     fig=pylab.figure(figsize=(6,4))
#     x = error
#     y = set
#     #to smooth the lines between the points
#     from scipy.interpolate import interp1d
#     x_new = np.linspace(x.min(), x.max(),500)
#     f = interp1d(x, y, kind='quadratic')
#     y_smooth=f(x_new)
#     plt.plot(x_new,y_smooth)
#     plt.scatter(x, y,linewidths=0)
#     #end
#     plt.plot([0, 4.8, 4.8], [60, 60, 0],'k--')
#     plt.plot([0, 9, 9], [90, 90, 0],'k--')
#     plt.ylabel('Percentage Data set [$\%$]')
#     plt.xlabel('Maximum absolute percentage error [$\%$]')
#     plt.xlim(0,35)
#     plt.ylim(0,100)
#     plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#            [r'0', r'10', r'20', r'30',r'40', r'50', r'60', r'70', r'80', r'90',r'100'])
#     plt.tight_layout(pad=0.2)        
#     plt.tick_params(direction='in')
#     #plt.show()
#     fig.savefig('ANN_Tmin_distribution.pdf') 
#     plt.close()
#     
#     ##########plot Tmin versus Bf/Bw for fixed P = 0.1 MPa, Tsub = 0 and L/D = 2.67, 5, 6.68 ##########
#     fig=pylab.figure(figsize=(6,4))
#     x = BfBw[np.array([298,318,338,342,346,350,367,368,369,370,375])] #BfBw[np.array([1,6,7,8,10])]
#     y1 = Tmin_exp[np.array([298,318,338,342,346,350,367,368,369,370,375])]
#     x2 = np.linspace(0, 0.6, num=50)
#     y2 = [256.3690185546875, 263.80029296875, 271.0977783203125, 278.25384521484375, 285.26080322265625, 292.1114196777344, 298.79962158203125, 305.3194885253906, 311.66558837890625, 317.8337707519531, 323.81976318359375, 329.6197509765625, 335.23126220703125, 340.65167236328125, 345.8790283203125, 350.9114990234375, 355.74859619140625, 360.389404296875, 364.833740234375, 369.08172607421875, 373.1337890625, 376.99090576171875, 380.65386962890625, 384.12420654296875, 387.4032897949219, 390.49322509765625, 393.39556884765625, 396.11279296875, 398.6470947265625, 401.00091552734375, 403.17681884765625, 405.17742919921875, 407.0052490234375, 408.6633605957031, 410.15460205078125, 411.48187255859375, 412.6478271484375, 413.655517578125, 414.50811767578125, 415.208251953125, 415.7589111328125, 416.1630554199219, 416.42388916015625, 416.5440673828125, 416.5263366699219, 416.37396240234375, 416.08917236328125, 415.6751708984375, 415.1343688964844, 414.4697570800781]
#     plt.plot(x,y1,'ro',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 2.67$')
#     #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='r')
#     plt.plot(x2,y2,'-r',markersize=5,markeredgewidth=0.1,alpha=0.9)
#     
#     x = BfBw[np.array([0,6,5,7,9,11,13,15,17,19,36])] #BfBw[np.array([1,6,7,8,10])]
#     y1 = Tmin_exp[np.array([0,6,5,7,9,11,13,15,17,19,36])]
#     x2 = np.linspace(0, 0.6, num=50)
#     y2 = [300.59295654296875, 306.8948669433594, 313.0576477050781, 319.0775146484375, 324.9509582519531, 330.6752624511719, 336.24755859375, 341.6661071777344, 346.92889404296875, 352.0347900390625, 356.983154296875, 361.7731628417969, 366.40447998046875, 370.8775634765625, 375.1927185058594, 379.3503723144531, 383.3516845703125, 387.197509765625, 390.88934326171875, 394.4284362792969, 397.81671142578125, 401.05609130859375, 404.1476745605469, 407.0942077636719, 409.8975524902344, 412.55975341796875, 415.08319091796875, 417.4701232910156, 419.72265625, 421.84332275390625, 423.83441162109375, 425.6979675292969, 427.43695068359375, 429.052978515625, 430.54840087890625, 431.92596435546875, 433.187744140625, 434.335693359375, 435.3721923828125, 436.29931640625, 437.11907958984375, 437.83343505859375, 438.44451904296875, 438.9541931152344, 439.3642578125, 439.6764831542969, 439.8926696777344, 440.0146484375, 440.0437927246094, 439.98187255859375]
#     plt.plot(x,y1,'bs',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 5$')
#     #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='b')
#     plt.plot(x2,y2,'-b',markersize=5,markeredgewidth=0.1,alpha=0.9)
#     
#     x = BfBw[np.array([194,250,278,252,253])] #BfBw[np.array([1,6,7,8,10])]
#     y1 = Tmin_exp[np.array([194,250,278,252,253])]
#     x2 = np.linspace(0, 0.6, num=50)
#     y2 = [347.0565490722656, 352.53314208984375, 357.86920166015625, 363.0633544921875, 368.11419677734375, 373.02056884765625, 377.78192138671875, 382.3980712890625, 386.86895751953125, 391.1947326660156, 395.37640380859375, 399.4144287109375, 403.3099060058594, 407.064453125, 410.67919921875, 414.1558837890625, 417.49609375, 420.7022705078125, 423.775634765625, 426.71881103515625, 429.5338134765625, 432.22283935546875, 434.78826904296875, 437.2324523925781, 439.5574951171875, 441.76593017578125, 443.85992431640625, 445.842041015625, 447.71435546875, 449.4796142578125, 451.1395263671875, 452.6968078613281, 454.15313720703125, 455.51129150390625, 456.77301025390625, 457.9405517578125, 459.01568603515625, 460.0006103515625, 460.89703369140625, 461.7070007324219, 462.4317626953125, 463.07366943359375, 463.6339111328125, 464.1144714355469, 464.516357421875, 464.84136962890625, 465.0908508300781, 465.26593017578125, 465.3681335449219, 465.3983154296875]
#     plt.plot(x,y1,'k^',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 6.68$')
#     #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
#     plt.plot(x2,y2,'-k',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'ANN')
#     
#     #plt.ylim(200,450)
#     #plt.xlim(0,0.2)
#     plt.xlabel(r'$\beta_f/\beta_w$ [$-$]')
#     plt.ylabel(r'$T_{min}$ [$\degree$C]')
#     leg = plt.legend(loc='best',fancybox=False,numpoints=1)
#     frame  = leg.get_frame()  
#     frame.set_linewidth(0.5)
#     plt.tight_layout()
#     #plt.show()
#     fig.savefig('ANN_Tmin_vary_LD.pdf')
#     plt.close()
#     
#     
#     ##########plot Tmin versus Bf/Bw for fixed P = 0.1Mpa, L/D = 2.67 ##########
#     fig=pylab.figure(figsize=(6,4))
#     x = Tsub[367:378]
#     y1 = Tmin_exp[367:378]#Tmin_exp[np.array([36,37,38,39,40,41,42])]
#     x2 = np.arange(0,41,1)
#     y2 = [223.1943817138672, 242.20481872558594, 259.4189758300781, 275.0093994140625, 289.15313720703125, 302.03411865234375, 313.8408508300781, 324.7634582519531, 334.9884033203125, 344.6937561035156, 354.0435791015625, 363.184326171875, 372.241455078125, 381.3172302246094, 390.4928894042969, 399.8265380859375, 409.3573303222656, 419.10687255859375, 429.0830078125, 439.28143310546875, 449.68914794921875, 460.28680419921875, 471.0513000488281, 481.956298828125, 492.9743957519531, 504.0788879394531, 515.2427978515625, 526.44091796875, 537.649658203125, 548.8475341796875, 560.0147094726562, 571.1333618164062, 582.18798828125, 593.1644287109375, 604.0509033203125, 614.837158203125, 625.5139770507812, 636.0740966796875, 646.5113525390625, 656.8206787109375, 666.998291015625]#Tmin_ANN[278:294]#Tmin_ANN[np.array([36,37,38,39,40,41,42])]
#     plt.plot(x,y1,'ro',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 2.67, \beta_f/\beta_w=0.032$ (Inconel)')
#     #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
#     plt.plot(x2,y2,'-r',markersize=5,markeredgewidth=0.1,alpha=0.9)#,label=r'ANN')
#     
#     x = Tsub[295:314]
#     y1 = Tmin_exp[295:314]
#     x2 = np.arange(0,41,1)
#     y2 = [250.77780151367188, 270.1280822753906, 287.5934143066406, 303.3346862792969, 317.5178527832031, 330.316650390625, 341.91033935546875, 352.4842529296875, 362.2234802246094, 371.30902099609375, 379.91241455078125, 388.1914978027344, 396.28619384765625, 404.315673828125, 412.37860107421875, 420.5509338378906, 428.888916015625, 437.43035888671875, 446.1964416503906, 455.19549560546875, 464.424072265625, 473.871337890625, 483.51947021484375, 493.3471374511719, 503.33074951171875, 513.444091796875, 523.662109375, 533.95947265625, 544.3128051757812, 554.6990356445312, 565.0975952148438, 575.4893798828125, 585.857177734375, 596.185302734375, 606.4605712890625, 616.6705322265625, 626.8045654296875, 636.8540649414062, 646.8114013671875, 656.6697387695312, 666.423583984375]#Tmin_ANN[355:366]
#     plt.plot(x,y1,'bs',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 2.67, \beta_f/\beta_w=0.053$ (Stainless steel)')
#     #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
#     plt.plot(x2,y2,'-b',markersize=5,markeredgewidth=0.1,alpha=0.9)#,label=r'ANN')
#     
#     x = Tsub[355:366]
#     y1 = Tmin_exp[355:366]
#     x2 = np.arange(0,41,1)
#     y2 = [275.4169616699219, 295.03271484375, 312.7061767578125, 328.589599609375, 342.8387756347656, 355.61578369140625, 367.0911560058594, 377.44073486328125, 386.84454345703125, 395.4812927246094, 403.5244140625, 411.13665771484375, 418.467041015625, 425.646240234375, 432.78619384765625, 439.97747802734375, 447.2912902832031, 454.7793884277344, 462.4768371582031, 470.4032287597656, 478.56573486328125, 486.9624328613281, 495.5823974609375, 504.4097900390625, 513.4249267578125, 522.6049194335938, 531.9263916015625, 541.36572265625, 550.8991088867188, 560.5042724609375, 570.1595458984375, 579.8450927734375, 589.54296875, 599.2362060546875, 608.91015625, 618.551513671875, 628.148193359375, 637.68994140625, 647.1676025390625, 656.5736083984375, 665.901123046875]#Tmin_ANN[355:366]
#     plt.plot(x,y1,'k^',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 2.67, \beta_f/\beta_w=0.066$ (Zirconium)')
#     #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
#     plt.plot(x2,y2,'-k',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'ANN')
#     
#     #plt.ylim(200,500)
#     plt.xlim(0,20)
#     plt.xlabel(r'$T_{sub}$ [$\degree$C]')
#     plt.ylabel(r'$T_{min}$ [$\degree$C]')
#     leg = plt.legend(loc='best',fancybox=False,numpoints=1)
#     frame  = leg.get_frame()  
#     frame.set_linewidth(0.5)
#     plt.tight_layout()
#     #plt.show()
#     fig.savefig('ANN_Tmin_vary_BfBw.pdf')
#     plt.close()
#     
#     ##########plot Tmin versus Tsub for fixed P = 0.1Mpa, L/D fixed = 6.5 ##########
#     fig=pylab.figure(figsize=(6,4))
#     x = Tsub[278:294]
#     y1 = Tmin_exp[278:294]
#     x2 = np.arange(0,41,1)
#     y2 = [354.20294189453125, 379.9591064453125, 403.31536865234375, 424.3001708984375, 443.0004577636719, 459.54840087890625, 474.108642578125, 486.8673095703125, 498.0201416015625, 507.765625, 516.298095703125, 523.8021240234375, 530.4503784179688, 536.40087890625, 541.7962646484375, 546.7633056640625, 551.4136352539062, 555.84375, 560.13671875, 564.3624267578125, 568.5796508789062, 572.8369750976562, 577.1728515625, 581.618408203125, 586.1973876953125, 590.9273681640625, 595.8201904296875, 600.884033203125, 606.12255859375, 611.536376953125, 617.123779296875, 622.8800048828125, 628.7996826171875, 634.8756103515625, 641.0985717773438, 647.46044921875, 653.950927734375, 660.559814453125, 667.27734375, 674.093017578125, 680.9962158203125,]
#     plt.plot(x,y1,'ro',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 6.5,  \beta_f/\beta_w=0.017$ (Carbon steel)')
#     #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
#     plt.plot(x2,y2,'-r',markersize=5,markeredgewidth=0.1,alpha=0.9)
#     
#     x = Tsub[270:277]
#     y1 = Tmin_exp[270:277]
#     x2 = np.arange(0,41,1)
#     y2 = [368.33197021484375, 394.46734619140625, 418.20330810546875, 439.5555419921875, 458.60052490234375, 475.46295166015625, 490.30279541015625, 503.3028564453125, 514.6585083007812, 524.568603515625, 533.2291870117188, 540.8275146484375, 547.539306640625, 553.5256958007812, 558.9324951171875, 563.8901977539062, 568.5130004882812, 572.90087890625, 577.1387939453125, 581.2998657226562, 585.4446411132812, 589.6234130859375, 593.8764038085938, 598.23583984375, 602.7268676757812, 607.36767578125, 612.1715698242188, 617.1466674804688, 622.2973022460938, 627.6246337890625, 633.126953125, 638.7996826171875, 644.6377563476562, 650.6334838867188, 656.7786865234375, 663.06396484375, 669.4798583984375, 676.0157470703125, 682.6615600585938, 689.4066772460938, 696.2408447265625,]
#     plt.plot(x,y1,'bs',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 6.5, \beta_f/\beta_w=0.053$ (Stainless steel)')
#     #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
#     plt.plot(x2,y2,'-b',markersize=5,markeredgewidth=0.1,alpha=0.9)
#            
#     x = Tsub[252:269]
#     y1 = Tmin_exp[252:269]
#     x2 = np.arange(0,41,1)
#     y2 = [392.51214599609375, 419.196044921875, 443.4951477050781, 465.4002380371094, 484.9674377441406, 502.3059387207031, 517.5642700195312, 530.918701171875, 542.5614013671875, 552.6915283203125, 561.506591796875, 569.1983642578125, 575.9471435546875, 581.919921875, 587.2685546875, 592.129150390625, 596.6219482421875, 600.852294921875, 604.910400390625, 608.8737182617188, 612.8070068359375, 616.763671875, 620.7879028320312, 624.914306640625, 629.170166015625, 633.576171875, 638.1463012695312, 642.8905639648438, 647.8145751953125, 652.9196166992188, 658.20458984375, 663.666259765625, 669.298828125, 675.0950927734375, 681.046875, 687.1446533203125, 693.3787841796875, 699.73876953125, 706.2139892578125, 712.7938232421875, 719.4673461914062,]
#     plt.plot(x,y1,'k^',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 6.5, \beta_f/\beta_w=0.119$ (Zirconium)')
#     #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
#     plt.plot(x2,y2,'-k',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'ANN')
#     
#     plt.ylim(200,700)
#     plt.xlim(0,20)
#     plt.xlabel(r'$T_{sub}$ [$\degree$C]')
#     plt.ylabel(r'$T_{min}$ [$\degree$C]')
#     leg = plt.legend(loc='best',fancybox=False,numpoints=1)
#     frame  = leg.get_frame()  
#     frame.set_linewidth(0.5)
#     plt.tight_layout()
#     #plt.show()
#     fig.savefig('ANN_Tmin_fixed_pressure.pdf')
#     plt.close()
# 
#     ##########plot Tmin versus Psat for fixed Tsub = 0, Bw/Bf= 0.014 (platimnum) vary L/D = 16.67, 25, 41.6##########
#     fig=pylab.figure(figsize=(6,4))
#     x = Psat[104:144]
#     y1 = Tmin_exp[104:144]
#     x2 = np.linspace(0, 3, num=100)
#     y2 = [186.7136993408203, 191.45130920410156, 195.8917999267578, 200.060302734375, 203.9804229736328, 207.67376708984375, 211.16036987304688, 214.4589385986328, 217.58644104003906, 220.55902099609375, 223.39083862304688, 226.0954132080078, 228.68484497070312, 231.17068481445312, 233.5629425048828, 235.87094116210938, 238.1033172607422, 240.26783752441406, 242.37164306640625, 244.4210662841797, 246.42184448242188, 248.3794403076172, 250.29832458496094, 252.18316650390625, 254.0377197265625, 255.86520385742188, 257.66912841796875, 259.4517822265625, 261.2162780761719, 262.96417236328125, 264.6980895996094, 266.4190979003906, 268.129150390625, 269.8296203613281, 271.52154541015625, 273.2060546875, 274.8839416503906, 276.5561218261719, 278.2231750488281, 279.8856506347656, 281.5439453125, 283.19879150390625, 284.8503112792969, 286.49847412109375, 288.14404296875, 289.78704833984375, 291.4272766113281, 293.0653991699219, 294.7010192871094, 296.3345947265625, 297.965576171875, 299.5946044921875, 301.22119140625, 302.8454895019531, 304.467529296875, 306.0872497558594, 307.7045593261719, 309.31964111328125, 310.9322814941406, 312.54248046875, 314.1502685546875, 315.7554016113281, 317.35809326171875, 318.9582824707031, 320.5559997558594, 322.15118408203125, 323.74371337890625, 325.33392333984375, 326.921630859375, 328.5069580078125, 330.0898132324219, 331.6705322265625, 333.24896240234375, 334.8252258300781, 336.39935302734375, 337.9716491699219, 339.5419006347656, 341.1104736328125, 342.677490234375, 344.2430419921875, 345.8073425292969, 347.37030029296875, 348.93255615234375, 350.49371337890625, 352.0542297363281, 353.6142578125, 355.1744079589844, 356.7342529296875, 358.294189453125, 359.8544616699219, 361.41571044921875, 362.9775085449219, 364.54052734375, 366.10491943359375, 367.6710205078125, 369.23876953125, 370.80877685546875, 372.3810119628906, 373.9559326171875, 375.53411865234375]#193.23045349121094, 199.26084899902344, 204.69850158691406, 209.59335327148438, 213.9940948486328, 217.94650268554688, 221.49407958984375, 224.6778564453125, 227.53640747070312, 230.10516357421875, 232.41769409179688, 234.50473022460938, 236.3945770263672, 238.11312866210938, 239.68429565429688, 241.1300048828125, 242.4699249267578, 243.72198486328125, 244.9026336669922, 246.02630615234375, 247.10635375976562, 248.1544952392578, 249.18128967285156, 250.19619750976562, 251.20755004882812, 252.22279357910156, 253.24822998046875, 254.28958129882812, 255.3518524169922, 256.439208984375, 257.555419921875, 258.703857421875, 259.8865661621094, 261.1062316894531, 262.36456298828125, 263.6627502441406, 265.00213623046875, 266.3835144042969, 267.807373046875, 269.27435302734375, 270.78424072265625, 272.3373718261719, 273.9331970214844, 275.571533203125, 277.25238037109375, 278.97479248046875, 280.73822021484375, 282.5421142578125, 284.38580322265625, 286.2680969238281, 288.188720703125, 290.14642333984375, 292.1405944824219, 294.1701354980469, 296.234130859375, 298.3320617675781, 300.4624328613281, 302.62469482421875, 304.81781005859375, 307.04095458984375, 309.2931213378906, 311.57354736328125, 313.88140869140625, 316.2156982421875, 318.5757751464844, 320.9604187011719, 323.36956787109375, 325.8018493652344, 328.2566833496094, 330.73370361328125, 333.23150634765625, 335.749755859375, 338.2880554199219, 340.845458984375, 343.42156982421875, 346.0156555175781, 348.6273193359375, 351.2554931640625, 353.900390625, 356.5612487792969, 359.23736572265625, 361.928466796875, 364.6341552734375, 367.353759765625, 370.0872802734375, 372.83392333984375, 375.5936279296875, 378.36566162109375, 381.1503601074219, 383.94647216796875, 386.7544250488281, 389.57366943359375, 392.404052734375, 395.2450256347656, 398.09649658203125, 400.958251953125, 403.83001708984375, 406.71160888671875, 409.6025085449219, 412.5028076171875]
#     plt.plot(x,y1,'ro',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 16.67, \beta_f/\beta_w=0.014$ (Platnium)')
#     #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
#     plt.plot(x2,y2,'-r',markersize=5,markeredgewidth=0.1,alpha=0.9)
#     
#     x = Psat[43:83]
#     y1 = Tmin_exp[43:83]
#     x2 = np.linspace(0, 3, num=100)
#     y2 = [198.00152587890625, 202.58798217773438, 206.8929443359375, 210.9400634765625, 214.75140380859375, 218.34786987304688, 221.74794006347656, 224.9694061279297, 228.0283966064453, 230.93966674804688, 233.71682739257812, 236.37298583984375, 238.91883850097656, 241.36561584472656, 243.72262573242188, 245.9986114501953, 248.2017822265625, 250.3395538330078, 252.41824340820312, 254.4441375732422, 256.4224853515625, 258.358154296875, 260.25604248046875, 262.11981201171875, 263.9532775878906, 265.7595520019531, 267.54180908203125, 269.30224609375, 271.0436706542969, 272.7677307128906, 274.47674560546875, 276.17181396484375, 277.8548583984375, 279.52691650390625, 281.18896484375, 282.84228515625, 284.48748779296875, 286.1255187988281, 287.7568054199219, 289.3819580078125, 291.0014343261719, 292.6157531738281, 294.2250061035156, 295.82958984375, 297.4297180175781, 299.0255126953125, 300.61712646484375, 302.2044982910156, 303.7882080078125, 305.36773681640625, 306.9434509277344, 308.51544189453125, 310.0833435058594, 311.6473388671875, 313.2076416015625, 314.76385498046875, 316.31622314453125, 317.86474609375, 319.40924072265625, 320.94970703125, 322.4862060546875, 324.0189208984375, 325.5475158691406, 327.072021484375, 328.5927429199219, 330.1095886230469, 331.62225341796875, 333.1312255859375, 334.6363830566406, 336.1376647949219, 337.63531494140625, 339.1293640136719, 340.61981201171875, 342.10687255859375, 343.5904846191406, 345.0709533691406, 346.54833984375, 348.0225830078125, 349.4942626953125, 350.96295166015625, 352.4293212890625, 353.8934020996094, 355.3551025390625, 356.81500244140625, 358.2731018066406, 359.72979736328125, 361.1847839355469, 362.63873291015625, 364.092041015625, 365.5445251464844, 366.9964599609375, 368.44818115234375, 369.9002685546875, 371.3524169921875, 372.80517578125, 374.25897216796875, 375.71380615234375, 377.170166015625, 378.6282043457031, 380.088134765625]
#     plt.plot(x,y1,'bs',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 25, \beta_f/\beta_w=0.014$ (Platnium)')
#     #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
#     plt.plot(x2,y2,'-b',markersize=5,markeredgewidth=0.1,alpha=0.9)
#            
#     x = Psat[84:98]
#     y1 = Tmin_exp[84:98]
#     x2 = np.linspace(0, 3, num=100)
#     y2 = [201.839111328125, 206.37599182128906, 210.636474609375, 214.64366149902344, 218.4195098876953, 221.98390197753906, 225.35543823242188, 228.5515899658203, 231.58786010742188, 234.47900390625, 237.2384490966797, 239.87828063964844, 242.40994262695312, 244.84368896484375, 247.1890869140625, 249.45448303222656, 251.64822387695312, 253.77688598632812, 255.84725952148438, 257.8653564453125, 259.836181640625, 261.76513671875, 263.6559753417969, 265.5129089355469, 267.3395690917969, 269.1390380859375, 270.9143371582031, 272.6676940917969, 274.4018859863281, 276.11834716796875, 277.81927490234375, 279.506103515625, 281.18048095703125, 282.84356689453125, 284.4962463378906, 286.1397705078125, 287.77447509765625, 289.40167236328125, 291.02191162109375, 292.635009765625, 294.242431640625, 295.84381103515625, 297.4398193359375, 299.03045654296875, 300.6162414550781, 302.19708251953125, 303.7733459472656, 305.3449401855469, 306.91192626953125, 308.4745788574219, 310.03271484375, 311.5865478515625, 313.1360168457031, 314.6807861328125, 316.22161865234375, 317.7576599121094, 319.2894287109375, 320.8164978027344, 322.3394775390625, 323.8580322265625, 325.37200927734375, 326.8813171386719, 328.38641357421875, 329.8870544433594, 331.38311767578125, 332.87493896484375, 334.36236572265625, 335.845458984375, 337.32421875, 338.798583984375, 340.26904296875, 341.7354431152344, 343.19775390625, 344.65618896484375, 346.1109313964844, 347.56207275390625, 349.00958251953125, 350.4537353515625, 351.89471435546875, 353.33245849609375, 354.7674255371094, 356.1998291015625, 357.6295166015625, 359.05670166015625, 360.48193359375, 361.9050598144531, 363.32647705078125, 364.746337890625, 366.1651611328125, 367.58282470703125, 368.9996643066406, 370.41595458984375, 371.83209228515625, 373.2481689453125, 374.6643981933594, 376.081298828125, 377.49884033203125, 378.917724609375, 380.33770751953125, 381.7596435546875]
#     plt.plot(x,y1,'k^',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 41.67, \beta_f/\beta_w=0.014$ (Platnium)')
#     #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
#     plt.plot(x2,y2,'-k',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'ANN')
#     
#     plt.ylim(150,400)
#     plt.xlim(0,2)
#     plt.xlabel(r'$P_{sat}$ [MPa]')
#     plt.ylabel(r'$T_{min}$ [$\degree$C]')
#     leg = plt.legend(loc='best',fancybox=False,numpoints=1)
#     frame  = leg.get_frame()  
#     frame.set_linewidth(0.5)
#     plt.tight_layout()
#     #plt.show()
#     fig.savefig('ANN_Tmin_vary_pressure_BfBw.pdf')
#     plt.close() 
#     
#      
#     ##########plot Tmin versus Psat for fixed Tsub = 0, vary Bw/Bf= 0.017, 0.053, 0.117 , vary L/D = 5.15 and 6.5##########
#     fig=pylab.figure(figsize=(6,4))
#     x11 = Psat[209:219]
#     x22 = Psat[222:251]
#     x = np.concatenate((x11,x22))
#     y11 = Tmin_exp[209:219]
#     y22 = Tmin_exp[222:251]
#     y1 = np.concatenate((y11,y22))
#     x2 = np.linspace(0, 3, num=100)
#     y2 = [292.84808349609375, 311.24920654296875, 329.87200927734375, 348.5887145996094, 367.2711181640625, 385.79254150390625, 404.0314025878906, 421.8735656738281, 439.2138671875, 455.95880126953125, 472.0273132324219, 487.3516845703125, 501.8782653808594, 515.5667724609375, 528.390625, 540.33544921875, 551.39892578125, 561.58837890625, 570.9210205078125, 579.4208984375, 587.1190185546875, 594.0513916015625, 600.2574462890625, 605.7799072265625, 610.662841796875, 614.951171875, 618.6904296875, 621.9251708984375, 624.698974609375, 627.0545043945312, 629.031982421875, 630.6702270507812, 632.0060424804688, 633.0738525390625, 633.9053955078125, 634.53125, 634.9789428710938, 635.2742919921875, 635.440673828125, 635.5, 635.4716796875, 635.3739013671875, 635.2230224609375, 635.03369140625, 634.8192749023438, 634.591552734375, 634.361572265625, 634.138427734375, 633.9307250976562, 633.746337890625, 633.5914306640625, 633.471923828125, 633.3931884765625, 633.359130859375, 633.3738403320312, 633.4403686523438, 633.561279296875, 633.7392578125, 633.9755859375, 634.2718505859375, 634.6292724609375, 635.0484619140625, 635.5302734375, 636.0745849609375, 636.6815185546875, 637.3509521484375, 638.082763671875, 638.8759765625, 639.7303466796875, 640.6450805664062, 641.619384765625, 642.652099609375, 643.7420654296875, 644.888427734375, 646.08984375, 647.3452758789062, 648.6531982421875, 650.0125732421875, 651.4219970703125, 652.8797607421875, 654.385009765625, 655.9357299804688, 657.531005859375, 659.1691284179688, 660.8488159179688, 662.5684814453125, 664.3270874023438, 666.122802734375, 667.954345703125, 669.8203735351562, 671.7193603515625, 673.6502075195312, 675.6114501953125, 677.6016845703125, 679.61962890625, 681.663818359375, 683.7332763671875, 685.8265380859375, 687.9423217773438, 690.0797119140625,]
#     plt.plot(x,y1,'ro',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 6.5, \beta_f/\beta_w=0.017$ (Carbon steel)')
#     #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
#     plt.plot(x2,y2,'-r',markersize=5,markeredgewidth=0.1,alpha=0.9)
#     
#     x = Psat[194:208]
#     y1 = Tmin_exp[194:208]
#     x2 = np.linspace(0, 3, num=100)
#     y2 = [308.01837158203125, 326.14520263671875, 344.45745849609375, 362.828369140625, 381.13092041015625, 399.24090576171875, 417.03936767578125, 434.4152526855469, 451.26690673828125, 467.50445556640625, 483.05084228515625, 497.8421936035156, 511.8286437988281, 524.9736938476562, 537.25390625, 548.6585693359375, 559.1875, 568.8511962890625, 577.668212890625, 585.6649780273438, 592.8736572265625, 599.3310546875, 605.0777587890625, 610.1571044921875, 614.6131591796875, 618.491455078125, 621.8369140625, 624.6941528320312, 627.1067504882812, 629.1165161132812, 630.7637939453125, 632.086669921875, 633.12109375, 633.9014282226562, 634.4588623046875, 634.822998046875, 635.021484375, 635.0789794921875, 635.018798828125, 634.8619384765625, 634.6277465820312, 634.333740234375, 633.99560546875, 633.6280517578125, 633.2435302734375, 632.8538818359375, 632.4691162109375, 632.0986328125, 631.7507934570312, 631.432373046875, 631.1500244140625, 630.9088134765625, 630.7138061523438, 630.569091796875, 630.478271484375, 630.4439697265625, 630.468994140625, 630.5550537109375, 630.703857421875, 630.916748046875, 631.1944580078125, 631.5374755859375, 631.946533203125, 632.42138671875, 632.9620361328125, 633.5680541992188, 634.239013671875, 634.9744262695312, 635.7733154296875, 636.6348876953125, 637.55810546875, 638.5418701171875, 639.585205078125, 640.6866455078125, 641.8450317382812, 643.0591430664062, 644.327392578125, 645.6484375, 647.0209350585938, 648.443359375, 649.914306640625, 651.4322509765625, 652.9957275390625, 654.6031494140625, 656.2529907226562, 657.9441528320312, 659.6748046875, 661.443359375, 663.248779296875, 665.0891723632812, 666.963623046875, 668.8701171875, 670.8079833984375, 672.7752685546875, 674.770751953125, 676.7932739257812, 678.8413696289062, 680.9136352539062, 683.0091552734375, 685.12646484375,]
#     plt.plot(x,y1,'bs',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 6.5, \beta_f/\beta_w=0.053$ (Stainless steel)')
#     #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
#     plt.plot(x2,y2,'-b',markersize=5,markeredgewidth=0.1,alpha=0.9)
#     
#     x = Psat[166:193]
#     y1 = Tmin_exp[166:193]
#     x2 = np.linspace(0, 3, num=100)
#     y2 = [333.38519287109375, 351.01251220703125, 368.76263427734375, 386.51129150390625, 404.1351318359375, 421.5145263671875, 438.53570556640625, 455.0934143066406, 471.0928039550781, 486.4506530761719, 501.0966491699219, 514.9734497070312, 528.0377197265625, 540.2589111328125, 551.6193237304688, 562.11279296875, 571.7435913085938, 580.5257568359375, 588.4810791015625, 595.63818359375, 602.0313720703125, 607.6986083984375, 612.6817626953125, 617.0240478515625, 620.7703857421875, 623.9656982421875, 626.655029296875, 628.88232421875, 630.6905517578125, 632.1209106445312, 633.212646484375, 634.0032958984375, 634.5281982421875, 634.820068359375, 634.909912109375, 634.8262939453125, 634.595703125, 634.2425537109375, 633.788818359375, 633.2550659179688, 632.6598510742188, 632.019775390625, 631.3504638671875, 630.665283203125, 629.9764404296875, 629.29541015625, 628.631591796875, 627.9935302734375, 627.3890380859375, 626.824951171875, 626.3070068359375, 625.8401489257812, 625.4290161132812, 625.0771484375, 624.7877197265625, 624.5630493164062, 624.4058227539062, 624.3172607421875, 624.2987060546875, 624.3511352539062, 624.4750366210938, 624.6708984375, 624.9387817382812, 625.2783203125, 625.689453125, 626.1712646484375, 626.7230224609375, 627.34423828125, 628.0337524414062, 628.790283203125, 629.6129150390625, 630.5001220703125, 631.4508056640625, 632.4633178710938, 633.5367431640625, 634.6688232421875, 635.8582763671875, 637.1038818359375, 638.4038696289062, 639.7567138671875, 641.160400390625, 642.6138916015625, 644.115234375, 645.6629638671875, 647.2554931640625, 648.8910522460938, 650.568359375, 652.2855224609375, 654.041259765625, 655.8336791992188, 657.661865234375, 659.5238037109375, 661.41845703125, 663.34375, 665.298828125, 667.2824096679688, 669.2926025390625, 671.3282470703125, 673.3880615234375, 675.470947265625,]
#     plt.plot(x,y1,'gv',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 6.5, \beta_f/\beta_w=0.117$ (Zirconium)')
#     #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
#     plt.plot(x2,y2,'-g',markersize=5,markeredgewidth=0.1,alpha=0.9)
#     
#     x = Psat[19:35]
#     y1 = Tmin_exp[19:35]
#     x2 = np.linspace(0, 3, num=100)
#     y2 = [295.9613037109375, 311.1558837890625, 327.0233154296875, 343.4775390625, 360.4207763671875, 377.74639892578125, 395.340087890625, 413.08251953125, 430.8524169921875, 448.5286865234375, 465.9933776855469, 483.1339416503906, 499.8457336425781, 516.0336303710938, 531.6138916015625, 546.51513671875, 560.6783447265625, 574.0579223632812, 586.621337890625, 598.3478393554688, 609.2286376953125, 619.2654418945312, 628.468994140625, 636.8583374023438, 644.4596557617188, 651.30419921875, 657.4278564453125, 662.8701171875, 667.6719970703125, 671.876708984375, 675.527587890625, 678.6680908203125, 681.3410034179688, 683.5882568359375, 685.449951171875, 686.9647827148438, 688.169677734375, 689.0992431640625, 689.785888671875, 690.2607421875, 690.5518798828125, 690.685546875, 690.6860961914062, 690.57568359375, 690.3748779296875, 690.102294921875, 689.7747192382812, 689.40771484375, 689.014892578125, 688.608642578125, 688.2005615234375, 687.8001708984375, 687.4166259765625, 687.057373046875, 686.729736328125, 686.4398193359375, 686.1924438476562, 685.9923706054688, 685.8436279296875, 685.7492065429688, 685.7125854492188, 685.7352294921875, 685.8193969726562, 685.9664916992188, 686.1774291992188, 686.4532470703125, 686.7940673828125, 687.200439453125, 687.672119140625, 688.2088623046875, 688.8104248046875, 689.4760131835938, 690.2050170898438, 690.996337890625, 691.8494262695312, 692.7627563476562, 693.7354736328125, 694.7662353515625, 695.8535766601562, 696.9962768554688, 698.1929931640625, 699.4422607421875, 700.7426147460938, 702.092529296875, 703.490478515625, 704.93505859375, 706.4246826171875, 707.9573974609375, 709.532470703125, 711.147705078125, 712.8017578125, 714.49365234375, 716.2210693359375, 717.98291015625, 719.77783203125, 721.6041259765625, 723.4605102539062, 725.3455200195312, 727.2578735351562, 729.196044921875,]
#     plt.plot(x,y1,'k^',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 5.15, \beta_f/\beta_w=0.117$ (Zirconium)')
#     #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
#     plt.plot(x2,y2,'-k',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'ANN')
#     
#     #plt.ylim(200,700)
#     plt.xlim(0,3)
#     plt.xlabel(r'$P_{sat}$ [MPa]')
#     plt.ylabel(r'$T_{min}$ [$\degree$C]')
#     leg = plt.legend(loc='best',fancybox=False,numpoints=1)
#     frame  = leg.get_frame()  
#     frame.set_linewidth(0.5)
#     plt.tight_layout()
#     #plt.show()
#     fig.savefig('ANN_Tmin_vary_pressure_LD.pdf')
#     plt.close()    
    
if __name__ == '__main__':
    
    Calculate()
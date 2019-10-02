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
    Tmin = float(data[i][0])
    Tsub = float(data[i][1])
    Psat = float(data[i][2])
    Mat = str(data[i][3])
    LD = float(data[i][4])
    Bf = float(data[i][5])
    Bw = float(data[i][6])
    BfBw = float(data[i][7])
    Tmin_Mori=float(data[i][8])
    Tmin_Adler=float(data[i][9])
    Tmin_Dhir=float(data[i][10])
    Tmin_Lauer=float(data[i][11])
    Tmin_Freud=float(data[i][12])
    Tmin_shikha_7=float(data[i][13])
    Tmin_shikha_8=float(data[i][14])
    Tmin_shikha_9=float(data[i][15])
    i=i+1
    
    while i < (end - start+1):
        Tmin = np.append(Tmin,float(data[i][0]))
        Tsub = np.append(Tsub,float(data[i][1]))
        Psat = np.append(Psat,float(data[i][2])) 
        Mat = np.append(Mat,str(data[i][3]))
        LD = np.append(LD,float(data[i][4]))
        Bf = np.append(Bf,float(data[i][5]))
        Bw = np.append(Bw,float(data[i][6]))
        BfBw = np.append(BfBw,float(data[i][7]))
        Tmin_Mori= np.append(Tmin_Mori,float(data[i][8]))
        Tmin_Adler=np.append(Tmin_Adler,float(data[i][9]))
        Tmin_Dhir=np.append(Tmin_Dhir,float(data[i][10]))
        Tmin_Lauer=np.append(Tmin_Lauer,float(data[i][11]))
        Tmin_Freud=np.append(Tmin_Freud,float(data[i][12]))
        Tmin_shikha_7=np.append(Tmin_shikha_7,float(data[i][13]))
        Tmin_shikha_8=np.append(Tmin_shikha_8,float(data[i][14]))
        Tmin_shikha_9=np.append(Tmin_shikha_9,float(data[i][15]))
        i=i+1
        Data = [Tmin,Tsub,Psat,Mat,LD,Bf,Bw,BfBw,Tmin_Mori,Tmin_Adler,Tmin_Dhir,Tmin_Lauer,Tmin_Freud,Tmin_shikha_7,Tmin_shikha_8,Tmin_shikha_9]
    
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
    end=379
    filename = 'Data_Collection.csv'
    
    #Define inputs
    [Tmin_exp,Tsub,Psat,Mat,LD,Bf,Bw,BfBw,Tmin_Mori,Tmin_Adler,Tmin_Dhir,Tmin_Lauer,Tmin_Freud,Tmin_shikha_7,Tmin_shikha_8,Tmin_shikha_9] = Import(start,end,filename)
    
    mode = 'run'
    
    from keras.models import Model,Sequential,load_model
    from keras.layers import Input,Flatten,Dense,LSTM,merge,Dropout,concatenate
    #from keras.engine import merge # from Keras version 1.2.2
    #from keras.layers.merge import concatenate
    from keras.layers import GaussianNoise
    from keras.utils import plot_model
    from keras.callbacks import TensorBoard
    from keras import regularizers
    
    
    #Normalize all parameters
    Tmin_exp_norm = Normalize(Tmin_exp, 206.8841, 727.8873239)
    Tsub_norm = Normalize(Tsub, 0, 39.84150546)
    Psat_norm = Normalize(Psat, 0.001185867, 3.003378378)
    LD_norm = Normalize(LD, 2.67, 63.5)
    Bf_norm = Normalize(Bf, 2428162.849, 2744290.164)
    Bw_norm = Normalize(Bw, 5168800, 1379121205)
    BfBw_norm = Normalize(BfBw, 0.001989845, 0.530923555)
    
    #convert to numpy array
    Tmin_exp_norm = np.array(Tmin_exp_norm)
    Tsub_norm = np.array(Tsub_norm)
    Psat_norm = np.array(Psat_norm)
    LD_norm = np.array(LD_norm)
    Bf_norm = np.array(Bf_norm)
    Bw_norm = np.array(Bw_norm)
    BfBw_norm = np.array(BfBw_norm)
    
    # split into input (X) and output (Y) variables
    X = np.column_stack((Tsub_norm, Psat_norm))
    X = np.column_stack((X, LD_norm))
    X = np.column_stack((X, BfBw_norm))
    Y = Tmin_exp_norm
    
    from sklearn.model_selection import train_test_split
    # shuffle the data before splitting for validation
    X_remain, X_valid, Y_remain, Y_valid = train_test_split(X, Y, test_size=0.15, shuffle= True)
    X_train, X_test, Y_train, Y_test = train_test_split(X_remain, Y_remain, test_size=0.15, shuffle= True)
    
    SC = np.array([]) #empty score array
    
    for i in range(1):
        if mode == 'training':
            # create model
            model = Sequential()
            model.add(Dense(i+12, input_dim=4, activation='tanh')) #init='uniform' #use_bias = True, bias_initializer='zero' #4 is perfect
            #model.add(GaussianNoise(0.1))
            #model.add(Dropout(0.2)) #Dropout is a technique where randomly selected neurons are ignored during training.
            model.add(Dense(i+12, activation='tanh'))
            #model.add(GaussianNoise(0.1))
            model.add(Dense(i+12, activation='tanh'))
            model.add(Dense(1, activation='linear'))
              
            plot_model(model, to_file='model.pdf',show_shapes=True,show_layer_names=True)
      
            # Compile model
            model.compile(optimizer='adamax',loss='mse',metrics=['mae',coeff_determination])
              
            # fit the model
            history = model.fit(X_train,
                                Y_train,
                                epochs=6000 , #Cut the epochs in half when using sequential 
                                batch_size=30, #increase the batch size results in faster compiler an d high error, while smaller batch size results in slower compiler and slightly accurate model
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
            fig.savefig('ANN_history_Tmin_loss.pdf')
    
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
            fig.savefig('ANN_history_Tmin_acc.pdf')
                    
            # Save the model
            model.save('ANN_model_Tmin_2.h5')
        
        elif mode == 'run':
        
            # Load the model
            model = load_model('ANN_model_Tmin.h5',custom_objects={'coeff_determination': coeff_determination})
            
        
        # Run the model
        Tmin_ANN = model.predict(X)
        Tmin_ANN = DeNormalize(Tmin_ANN.reshape(-1), 206.8841, 727.8873239) #W = DeNormalize(W.reshape(-1),1000,8000)
        
        # evaluate the model (for the last batch)
        scores = model.evaluate(X,Y)
        SC = np.append(SC,scores[1]*100)
        print('')
        print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
           
        # extract the weight and bias
        weights = model.layers[3].get_weights()[0]
        biases = model.layers[3].get_weights()[1]
        
        #to chnage the percision of printed numbers
        np.set_printoptions(precision=4, suppress=True,
                       threshold=10000,
                       linewidth=150)
        print('')
        print 'weights = ', weights.transpose()
        print 'biases = ', biases

        # Save the architecture of a model, and not its weights or its training configuration
        # save as JSON
        # json_string = model.to_json()
        
        # save as YAML
        # yaml_string = model.to_yaml()
    print('')
    for i in range(len(SC)):
        print (SC[i])    
        
    # to SAVE into excel file
#     for i in range(0,(end-start+1)):
#  
#  
#         data_calc = {'Tmin':[Tmin_ANN[i]]}  #data_calc = {'Tdis':[T[i]],'mdot':[Mref[i]],'mdot_inj':[Minj[i]], 'Wdot':[W[i]],'etaoa':[eta_s[i]],'fq':[Q[i]/W[i]]} 
#              
#          
#         # Write to Excel
#         filename = os.path.dirname(__file__)+'/Tmin_output.xlsx'
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
    sep_val = 0.15
    n_len = len(Tmin_ANN)
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
    
    print 'Method:','MSE','MAE','MAPE','Rsquared'
    print 'Tmin_ANN:',mse(Tmin_ANN,Tmin_exp),REmean(Tmin_exp,Tmin_ANN),mape(Tmin_ANN, Tmin_exp),Rsquared(Tmin_exp,Tmin_ANN)*100
    print 'Tmin_Mori:',mse(Tmin_Mori,Tmin_exp),REmean(Tmin_exp,Tmin_Mori),mape(Tmin_Mori, Tmin_exp),Rsquared(Tmin_exp,Tmin_Mori)*100
    print 'Tmin_Adler:',mse(Tmin_Adler,Tmin_exp),REmean(Tmin_exp,Tmin_Adler),mape(Tmin_Adler, Tmin_exp),Rsquared(Tmin_exp,Tmin_Adler)*100
    print 'Tmin_Dhir:',mse(Tmin_Dhir,Tmin_exp),REmean(Tmin_exp,Tmin_Dhir),mape(Tmin_Dhir, Tmin_exp),Rsquared(Tmin_exp,Tmin_Dhir)*100
    print 'Tmin_Lauer:',mse(Tmin_Lauer,Tmin_exp),REmean(Tmin_exp,Tmin_Lauer),mape(Tmin_Lauer, Tmin_exp),Rsquared(Tmin_exp,Tmin_Lauer)*100
    print 'Tmin_Freud:',mse(Tmin_Freud,Tmin_exp),REmean(Tmin_exp,Tmin_Freud),mape(Tmin_Freud, Tmin_exp),Rsquared(Tmin_exp,Tmin_Freud)*100
    print 'Tmin_shikha_7:',mse(Tmin_shikha_7,Tmin_exp),REmean(Tmin_exp,Tmin_shikha_7),mape(Tmin_shikha_7, Tmin_exp),Rsquared(Tmin_exp,Tmin_shikha_7)*100
    print 'Tmin_shikha_8:',mse(Tmin_shikha_8,Tmin_exp),REmean(Tmin_exp,Tmin_shikha_8),mape(Tmin_shikha_8, Tmin_exp),Rsquared(Tmin_exp,Tmin_shikha_8)*100
    print 'Tmin_shikha_9:',mse(Tmin_shikha_9,Tmin_exp),REmean(Tmin_exp,Tmin_shikha_9),mape(Tmin_shikha_9, Tmin_exp),Rsquared(Tmin_exp,Tmin_shikha_9)*100
#     for i in range(len(Tmin_ANN)): # print the measure absolute error (%) for all data
#         print (REmean(Tmin_exp[i],Tmin_ANN[i])*100)
    error = np.array([0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
    set = np.array([0,13.72031662,27.44063325,37.73087071,49.34036939,59.10290237,67.01846966,73.08707124,78.89182058,82.58575198,85.75197889,95.25065963,97.09762533,98.1530343,98.1530343,98.1530343,98.41688654,98.41688654,98.41688654,98.41688654,98.41688654,98.94459103,99.20844327,99.47229551,99.73614776,99.73614776,99.73614776,100,100])

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
    Tmin_ANN_valid = model.predict(X_valid)
    Tmin_ANN_valid = DeNormalize(Tmin_ANN_valid.reshape(-1), 206.8841, 727.8873239)
    Tmin_exp_valid = DeNormalize(Y_valid.reshape(-1), 206.8841, 727.8873239) 
    #Training
    Tmin_ANN_train = model.predict(X_train)
    Tmin_ANN_train = DeNormalize(Tmin_ANN_train.reshape(-1), 206.8841, 727.8873239)
    Tmin_exp_train = DeNormalize(Y_train.reshape(-1), 206.8841, 727.8873239) 
    #Testing
    Tmin_ANN_test = model.predict(X_test)
    Tmin_ANN_test = DeNormalize(Tmin_ANN_test.reshape(-1), 206.8841, 727.8873239)
    Tmin_exp_test = DeNormalize(Y_test.reshape(-1), 206.8841, 727.8873239) 
        
    # plot all data
    fig=pylab.figure(figsize=(4,4))
    plt.plot(Tmin_ANN_train,Tmin_exp_train,'ro',ms = 3,mec='black',mew=0.5,label='Training points')
    plt.plot(Tmin_ANN_test,Tmin_exp_test,'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
    plt.plot(Tmin_ANN_valid,Tmin_exp_valid,'g^',ms = 4,mec='black',mew=0.5,label='Validation points')
    plt.text(550,200,'R$^2$ = {:0.01f}%\n'.format(Rsquared(Tmin_exp,Tmin_ANN)*100)+'MAE = {:0.01f}%\n'.format(mape(Tmin_ANN,Tmin_exp))+'RMSE = {:0.01f}%\n'.format(rmse(Tmin_ANN,Tmin_exp)),ha='left',va='center',fontsize = 8)
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
    #plt.show()
    fig.savefig('ANN_Tmin_all.pdf')
    plt.close()
    
    # ICERD plots (training)
    fig=pylab.figure(figsize=(4,4))
    plt.plot(Tmin_ANN_train,Tmin_exp_train,'ro',ms = 3,mec='black',mew=0.5,label='Training points')
    plt.text(550,200,'R$^2$ = {:0.01f}%\n'.format(Rsquared(Tmin_exp_train,Tmin_ANN_train)*100)+'MAE = {:0.01f}%\n'.format(mape(Tmin_ANN_train,Tmin_exp_train))+'RMSE = {:0.01f}%\n'.format(rmse(Tmin_ANN_train,Tmin_exp_train)),ha='left',va='center',fontsize = 8)
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
    #plt.show()
    fig.savefig('ANN_Tmin_training.pdf')  
    plt.close()
    
    # ICERD plots (testing)
    fig=pylab.figure(figsize=(4,4))
    plt.plot(Tmin_ANN_test,Tmin_exp_test,'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
    plt.text(550,200,'R$^2$ = {:0.01f}%\n'.format(Rsquared(Tmin_exp_test,Tmin_ANN_test)*100)+'MAE = {:0.01f}%\n'.format(mape(Tmin_ANN_test,Tmin_exp_test))+'RMSE = {:0.01f}%\n'.format(rmse(Tmin_ANN_test,Tmin_exp_test)),ha='left',va='center',fontsize = 8)
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
    #plt.show()
    fig.savefig('ANN_Tmin_testing.pdf')
    plt.close()
    
    # ICERD plots (validation)
    fig=pylab.figure(figsize=(4,4))
    plt.plot(Tmin_ANN_valid,Tmin_exp_valid,'g^',ms = 4,mec='black',mew=0.5,label='Validation points')
    plt.text(550,200,'R$^2$ = {:0.01f}%\n'.format(Rsquared(Tmin_exp_valid,Tmin_ANN_valid)*100)+'MAE = {:0.01f}%\n'.format(mape(Tmin_ANN_valid,Tmin_exp_valid))+'RMSE = {:0.01f}%\n'.format(rmse(Tmin_ANN_valid,Tmin_exp_valid)),ha='left',va='center',fontsize = 8)
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
    #plt.show()
    fig.savefig('ANN_Tmin_validation.pdf') 
    plt.close()
    
    # Cumulative Distribution of all data versus maximum absolute error
    fig=pylab.figure(figsize=(6,4))
    x = error
    y = set
    #to smooth the lines between the points
    from scipy.interpolate import interp1d
    x_new = np.linspace(x.min(), x.max(),500)
    f = interp1d(x, y, kind='quadratic')
    y_smooth=f(x_new)
    plt.plot(x_new,y_smooth)
    plt.scatter(x, y,linewidths=0)
    #end
    plt.plot([0, 5.1, 5.1], [60, 60, 0],'k--')
    plt.plot([0, 11.5, 11.5], [90, 90, 0],'k--')
    plt.ylabel('Percentage Data set [$\%$]')
    plt.xlabel('Maximum absolute percentage error [$\%$]')
    plt.xlim(0,35)
    plt.ylim(0,100)
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
           [r'0', r'10', r'20', r'30',r'40', r'50', r'60', r'70', r'80', r'90',r'100'])
    plt.tight_layout(pad=0.2)        
    plt.tick_params(direction='in')
    #plt.show()
    fig.savefig('ANN_Tmin_distribution.pdf') 
    plt.close()
    
    ##########plot Tmin versus Bf/Bw for fixed P = 0.1 MPa, Tsub = 0 and L/D = 2.67, 5, 6.68 ##########
    fig=pylab.figure(figsize=(6,4))
    x = BfBw[np.array([298,318,338,342,346,350,367,368,369,370,375])] #BfBw[np.array([1,6,7,8,10])]
    y1 = Tmin_exp[np.array([298,318,338,342,346,350,367,368,369,370,375])]
    x2 = np.linspace(0, 0.6, num=50)
    y2 = [256.3690185546875, 263.80029296875, 271.0977783203125, 278.25384521484375, 285.26080322265625, 292.1114196777344, 298.79962158203125, 305.3194885253906, 311.66558837890625, 317.8337707519531, 323.81976318359375, 329.6197509765625, 335.23126220703125, 340.65167236328125, 345.8790283203125, 350.9114990234375, 355.74859619140625, 360.389404296875, 364.833740234375, 369.08172607421875, 373.1337890625, 376.99090576171875, 380.65386962890625, 384.12420654296875, 387.4032897949219, 390.49322509765625, 393.39556884765625, 396.11279296875, 398.6470947265625, 401.00091552734375, 403.17681884765625, 405.17742919921875, 407.0052490234375, 408.6633605957031, 410.15460205078125, 411.48187255859375, 412.6478271484375, 413.655517578125, 414.50811767578125, 415.208251953125, 415.7589111328125, 416.1630554199219, 416.42388916015625, 416.5440673828125, 416.5263366699219, 416.37396240234375, 416.08917236328125, 415.6751708984375, 415.1343688964844, 414.4697570800781]
    plt.plot(x,y1,'ro',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 2.67$')
    #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='r')
    plt.plot(x2,y2,'-r',markersize=5,markeredgewidth=0.1,alpha=0.9)
    
    x = BfBw[np.array([0,6,5,7,9,11,13,15,17,19,36])] #BfBw[np.array([1,6,7,8,10])]
    y1 = Tmin_exp[np.array([0,6,5,7,9,11,13,15,17,19,36])]
    x2 = np.linspace(0, 0.6, num=50)
    y2 = [300.59295654296875, 306.8948669433594, 313.0576477050781, 319.0775146484375, 324.9509582519531, 330.6752624511719, 336.24755859375, 341.6661071777344, 346.92889404296875, 352.0347900390625, 356.983154296875, 361.7731628417969, 366.40447998046875, 370.8775634765625, 375.1927185058594, 379.3503723144531, 383.3516845703125, 387.197509765625, 390.88934326171875, 394.4284362792969, 397.81671142578125, 401.05609130859375, 404.1476745605469, 407.0942077636719, 409.8975524902344, 412.55975341796875, 415.08319091796875, 417.4701232910156, 419.72265625, 421.84332275390625, 423.83441162109375, 425.6979675292969, 427.43695068359375, 429.052978515625, 430.54840087890625, 431.92596435546875, 433.187744140625, 434.335693359375, 435.3721923828125, 436.29931640625, 437.11907958984375, 437.83343505859375, 438.44451904296875, 438.9541931152344, 439.3642578125, 439.6764831542969, 439.8926696777344, 440.0146484375, 440.0437927246094, 439.98187255859375]
    plt.plot(x,y1,'bs',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 5$')
    #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='b')
    plt.plot(x2,y2,'-b',markersize=5,markeredgewidth=0.1,alpha=0.9)
    
    x = BfBw[np.array([194,250,278,252,253])] #BfBw[np.array([1,6,7,8,10])]
    y1 = Tmin_exp[np.array([194,250,278,252,253])]
    x2 = np.linspace(0, 0.6, num=50)
    y2 = [347.0565490722656, 352.53314208984375, 357.86920166015625, 363.0633544921875, 368.11419677734375, 373.02056884765625, 377.78192138671875, 382.3980712890625, 386.86895751953125, 391.1947326660156, 395.37640380859375, 399.4144287109375, 403.3099060058594, 407.064453125, 410.67919921875, 414.1558837890625, 417.49609375, 420.7022705078125, 423.775634765625, 426.71881103515625, 429.5338134765625, 432.22283935546875, 434.78826904296875, 437.2324523925781, 439.5574951171875, 441.76593017578125, 443.85992431640625, 445.842041015625, 447.71435546875, 449.4796142578125, 451.1395263671875, 452.6968078613281, 454.15313720703125, 455.51129150390625, 456.77301025390625, 457.9405517578125, 459.01568603515625, 460.0006103515625, 460.89703369140625, 461.7070007324219, 462.4317626953125, 463.07366943359375, 463.6339111328125, 464.1144714355469, 464.516357421875, 464.84136962890625, 465.0908508300781, 465.26593017578125, 465.3681335449219, 465.3983154296875]
    plt.plot(x,y1,'k^',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 6.68$')
    #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
    plt.plot(x2,y2,'-k',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'ANN')
    
    #plt.ylim(200,450)
    #plt.xlim(0,0.2)
    plt.xlabel(r'$\beta_f/\beta_w$ [$-$]')
    plt.ylabel(r'$T_{min}$ [$\degree$C]')
    leg = plt.legend(loc='best',fancybox=False,numpoints=1)
    frame  = leg.get_frame()  
    frame.set_linewidth(0.5)
    plt.tight_layout()
    #plt.show()
    fig.savefig('ANN_Tmin_vary_LD.pdf')
    plt.close()
    
    
    ##########plot Tmin versus Bf/Bw for fixed P = 0.1Mpa, L/D = 2.67 ##########
    fig=pylab.figure(figsize=(6,4))
    x = Tsub[367:378]
    y1 = Tmin_exp[367:378]#Tmin_exp[np.array([36,37,38,39,40,41,42])]
    x2 = np.arange(0,41,1)
    y2 = [223.1943817138672, 242.20481872558594, 259.4189758300781, 275.0093994140625, 289.15313720703125, 302.03411865234375, 313.8408508300781, 324.7634582519531, 334.9884033203125, 344.6937561035156, 354.0435791015625, 363.184326171875, 372.241455078125, 381.3172302246094, 390.4928894042969, 399.8265380859375, 409.3573303222656, 419.10687255859375, 429.0830078125, 439.28143310546875, 449.68914794921875, 460.28680419921875, 471.0513000488281, 481.956298828125, 492.9743957519531, 504.0788879394531, 515.2427978515625, 526.44091796875, 537.649658203125, 548.8475341796875, 560.0147094726562, 571.1333618164062, 582.18798828125, 593.1644287109375, 604.0509033203125, 614.837158203125, 625.5139770507812, 636.0740966796875, 646.5113525390625, 656.8206787109375, 666.998291015625]#Tmin_ANN[278:294]#Tmin_ANN[np.array([36,37,38,39,40,41,42])]
    plt.plot(x,y1,'ro',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 2.67, \beta_f/\beta_w=0.032$ (Inconel)')
    #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
    plt.plot(x2,y2,'-r',markersize=5,markeredgewidth=0.1,alpha=0.9)#,label=r'ANN')
    
    x = Tsub[295:314]
    y1 = Tmin_exp[295:314]
    x2 = np.arange(0,41,1)
    y2 = [250.77780151367188, 270.1280822753906, 287.5934143066406, 303.3346862792969, 317.5178527832031, 330.316650390625, 341.91033935546875, 352.4842529296875, 362.2234802246094, 371.30902099609375, 379.91241455078125, 388.1914978027344, 396.28619384765625, 404.315673828125, 412.37860107421875, 420.5509338378906, 428.888916015625, 437.43035888671875, 446.1964416503906, 455.19549560546875, 464.424072265625, 473.871337890625, 483.51947021484375, 493.3471374511719, 503.33074951171875, 513.444091796875, 523.662109375, 533.95947265625, 544.3128051757812, 554.6990356445312, 565.0975952148438, 575.4893798828125, 585.857177734375, 596.185302734375, 606.4605712890625, 616.6705322265625, 626.8045654296875, 636.8540649414062, 646.8114013671875, 656.6697387695312, 666.423583984375]#Tmin_ANN[355:366]
    plt.plot(x,y1,'bs',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 2.67, \beta_f/\beta_w=0.053$ (Stainless steel)')
    #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
    plt.plot(x2,y2,'-b',markersize=5,markeredgewidth=0.1,alpha=0.9)#,label=r'ANN')
    
    x = Tsub[355:366]
    y1 = Tmin_exp[355:366]
    x2 = np.arange(0,41,1)
    y2 = [275.4169616699219, 295.03271484375, 312.7061767578125, 328.589599609375, 342.8387756347656, 355.61578369140625, 367.0911560058594, 377.44073486328125, 386.84454345703125, 395.4812927246094, 403.5244140625, 411.13665771484375, 418.467041015625, 425.646240234375, 432.78619384765625, 439.97747802734375, 447.2912902832031, 454.7793884277344, 462.4768371582031, 470.4032287597656, 478.56573486328125, 486.9624328613281, 495.5823974609375, 504.4097900390625, 513.4249267578125, 522.6049194335938, 531.9263916015625, 541.36572265625, 550.8991088867188, 560.5042724609375, 570.1595458984375, 579.8450927734375, 589.54296875, 599.2362060546875, 608.91015625, 618.551513671875, 628.148193359375, 637.68994140625, 647.1676025390625, 656.5736083984375, 665.901123046875]#Tmin_ANN[355:366]
    plt.plot(x,y1,'k^',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 2.67, \beta_f/\beta_w=0.066$ (Zirconium)')
    #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
    plt.plot(x2,y2,'-k',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'ANN')
    
    #plt.ylim(200,500)
    plt.xlim(0,20)
    plt.xlabel(r'$T_{sub}$ [$\degree$C]')
    plt.ylabel(r'$T_{min}$ [$\degree$C]')
    leg = plt.legend(loc='best',fancybox=False,numpoints=1)
    frame  = leg.get_frame()  
    frame.set_linewidth(0.5)
    plt.tight_layout()
    #plt.show()
    fig.savefig('ANN_Tmin_vary_BfBw.pdf')
    plt.close()
    
    ##########plot Tmin versus Tsub for fixed P = 0.1Mpa, L/D fixed = 6.5 ##########
    fig=pylab.figure(figsize=(6,4))
    x = Tsub[278:294]
    y1 = Tmin_exp[278:294]
    x2 = np.arange(0,41,1)
    y2 = [350.39178466796875, 380.4754638671875, 407.14215087890625, 430.693359375, 451.3969421386719, 469.5000305175781, 485.2388916015625, 498.847412109375, 510.56097412109375, 520.6187744140625, 529.2615966796875, 536.7294921875, 543.257080078125, 549.0672607421875, 554.3673095703125, 559.343505859375, 564.1580810546875, 568.947509765625, 573.822265625, 578.8672485351562, 584.1439819335938, 589.6934814453125, 595.5386962890625, 601.6884765625, 608.139892578125, 614.8809204101562, 621.893798828125, 629.156005859375, 636.6423950195312, 644.3270263671875, 652.1832275390625, 660.1851806640625, 668.30810546875, 676.5280151367188, 684.822998046875, 693.1729125976562, 701.5595703125, 709.9655151367188, 718.3761596679688, 726.7775268554688, 735.1578369140625]
    plt.plot(x,y1,'ro',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 6.5,  \beta_f/\beta_w=0.017$ (Carbon steel)')
    #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
    plt.plot(x2,y2,'-r',markersize=5,markeredgewidth=0.1,alpha=0.9)
    
    x = Tsub[270:277]
    y1 = Tmin_exp[270:277]
    x2 = np.arange(0,41,1)
    y2 = [366.1865234375, 396.7049865722656, 423.78955078125, 447.7373046875, 468.80938720703125, 487.2441711425781, 503.2665100097656, 517.09765625, 528.9603271484375, 539.0809326171875, 547.690185546875, 555.0204467773438, 561.3016357421875, 566.7568359375, 571.596923828125, 576.0157470703125, 580.185791015625, 584.256591796875, 588.352294921875, 592.5723876953125, 596.99267578125, 601.6673583984375, 606.6311645507812, 611.9031982421875, 617.4893188476562, 623.3841552734375, 629.5758056640625, 636.0453491210938, 642.7708129882812, 649.7280883789062, 656.8917236328125, 664.235595703125, 671.7353515625, 679.3663940429688, 687.1063232421875, 694.9334106445312, 702.8284912109375, 710.773193359375, 718.75146484375, 726.7482299804688, 734.750244140625]
    plt.plot(x,y1,'bs',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 6.5, \beta_f/\beta_w=0.053$ (Stainless steel)')
    #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
    plt.plot(x2,y2,'-b',markersize=5,markeredgewidth=0.1,alpha=0.9)
           
    x = Tsub[252:269]
    y1 = Tmin_exp[252:269]
    x2 = np.arange(0,41,1)
    y2 = [397.76397705078125, 428.91656494140625, 456.6412048339844, 481.2353515625, 502.9532775878906, 522.0198974609375, 538.640625, 553.0125732421875, 565.3302001953125, 575.7902221679688, 584.5946655273438, 591.9495849609375, 598.064208984375, 603.1470947265625, 607.4029541015625, 611.02734375, 614.2022705078125, 617.09375, 619.8472900390625, 622.587646484375, 625.4178466796875, 628.4188232421875, 631.65234375, 635.1612548828125, 638.972900390625, 643.1016845703125, 647.550048828125, 652.3123779296875, 657.3768310546875, 662.7265014648438, 668.3411254882812, 674.198974609375, 680.277099609375, 686.55224609375, 693.0018310546875, 699.6038818359375, 706.337646484375, 713.1837158203125, 720.1236572265625, 727.1406860351562, 734.2195434570312]
    plt.plot(x,y1,'k^',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 6.5, \beta_f/\beta_w=0.119$ (Zirconium)')
    #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
    plt.plot(x2,y2,'-k',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'ANN')
    
    #plt.ylim(200,700)
    plt.xlim(0,20)
    plt.xlabel(r'$T_{sub}$ [$\degree$C]')
    plt.ylabel(r'$T_{min}$ [$\degree$C]')
    leg = plt.legend(loc='best',fancybox=False,numpoints=1)
    frame  = leg.get_frame()  
    frame.set_linewidth(0.5)
    plt.tight_layout()
    #plt.show()
    fig.savefig('ANN_Tmin_fixed_pressure.pdf')
    plt.close()

    ##########plot Tmin versus Psat for fixed Tsub = 0, Bw/Bf= 0.014 (platimnum) vary L/D = 16.67, 25, 41.6##########
    fig=pylab.figure(figsize=(6,4))
    x = Psat[104:144]
    y1 = Tmin_exp[104:144]
    x2 = np.linspace(0, 3, num=100)
    y2 = [186.7136993408203, 191.45130920410156, 195.8917999267578, 200.060302734375, 203.9804229736328, 207.67376708984375, 211.16036987304688, 214.4589385986328, 217.58644104003906, 220.55902099609375, 223.39083862304688, 226.0954132080078, 228.68484497070312, 231.17068481445312, 233.5629425048828, 235.87094116210938, 238.1033172607422, 240.26783752441406, 242.37164306640625, 244.4210662841797, 246.42184448242188, 248.3794403076172, 250.29832458496094, 252.18316650390625, 254.0377197265625, 255.86520385742188, 257.66912841796875, 259.4517822265625, 261.2162780761719, 262.96417236328125, 264.6980895996094, 266.4190979003906, 268.129150390625, 269.8296203613281, 271.52154541015625, 273.2060546875, 274.8839416503906, 276.5561218261719, 278.2231750488281, 279.8856506347656, 281.5439453125, 283.19879150390625, 284.8503112792969, 286.49847412109375, 288.14404296875, 289.78704833984375, 291.4272766113281, 293.0653991699219, 294.7010192871094, 296.3345947265625, 297.965576171875, 299.5946044921875, 301.22119140625, 302.8454895019531, 304.467529296875, 306.0872497558594, 307.7045593261719, 309.31964111328125, 310.9322814941406, 312.54248046875, 314.1502685546875, 315.7554016113281, 317.35809326171875, 318.9582824707031, 320.5559997558594, 322.15118408203125, 323.74371337890625, 325.33392333984375, 326.921630859375, 328.5069580078125, 330.0898132324219, 331.6705322265625, 333.24896240234375, 334.8252258300781, 336.39935302734375, 337.9716491699219, 339.5419006347656, 341.1104736328125, 342.677490234375, 344.2430419921875, 345.8073425292969, 347.37030029296875, 348.93255615234375, 350.49371337890625, 352.0542297363281, 353.6142578125, 355.1744079589844, 356.7342529296875, 358.294189453125, 359.8544616699219, 361.41571044921875, 362.9775085449219, 364.54052734375, 366.10491943359375, 367.6710205078125, 369.23876953125, 370.80877685546875, 372.3810119628906, 373.9559326171875, 375.53411865234375]#193.23045349121094, 199.26084899902344, 204.69850158691406, 209.59335327148438, 213.9940948486328, 217.94650268554688, 221.49407958984375, 224.6778564453125, 227.53640747070312, 230.10516357421875, 232.41769409179688, 234.50473022460938, 236.3945770263672, 238.11312866210938, 239.68429565429688, 241.1300048828125, 242.4699249267578, 243.72198486328125, 244.9026336669922, 246.02630615234375, 247.10635375976562, 248.1544952392578, 249.18128967285156, 250.19619750976562, 251.20755004882812, 252.22279357910156, 253.24822998046875, 254.28958129882812, 255.3518524169922, 256.439208984375, 257.555419921875, 258.703857421875, 259.8865661621094, 261.1062316894531, 262.36456298828125, 263.6627502441406, 265.00213623046875, 266.3835144042969, 267.807373046875, 269.27435302734375, 270.78424072265625, 272.3373718261719, 273.9331970214844, 275.571533203125, 277.25238037109375, 278.97479248046875, 280.73822021484375, 282.5421142578125, 284.38580322265625, 286.2680969238281, 288.188720703125, 290.14642333984375, 292.1405944824219, 294.1701354980469, 296.234130859375, 298.3320617675781, 300.4624328613281, 302.62469482421875, 304.81781005859375, 307.04095458984375, 309.2931213378906, 311.57354736328125, 313.88140869140625, 316.2156982421875, 318.5757751464844, 320.9604187011719, 323.36956787109375, 325.8018493652344, 328.2566833496094, 330.73370361328125, 333.23150634765625, 335.749755859375, 338.2880554199219, 340.845458984375, 343.42156982421875, 346.0156555175781, 348.6273193359375, 351.2554931640625, 353.900390625, 356.5612487792969, 359.23736572265625, 361.928466796875, 364.6341552734375, 367.353759765625, 370.0872802734375, 372.83392333984375, 375.5936279296875, 378.36566162109375, 381.1503601074219, 383.94647216796875, 386.7544250488281, 389.57366943359375, 392.404052734375, 395.2450256347656, 398.09649658203125, 400.958251953125, 403.83001708984375, 406.71160888671875, 409.6025085449219, 412.5028076171875]
    plt.plot(x,y1,'ro',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 16.67, \beta_f/\beta_w=0.014$ (Platnium)')
    #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
    plt.plot(x2,y2,'-r',markersize=5,markeredgewidth=0.1,alpha=0.9)
    
    x = Psat[43:83]
    y1 = Tmin_exp[43:83]
    x2 = np.linspace(0, 3, num=100)
    y2 = [198.00152587890625, 202.58798217773438, 206.8929443359375, 210.9400634765625, 214.75140380859375, 218.34786987304688, 221.74794006347656, 224.9694061279297, 228.0283966064453, 230.93966674804688, 233.71682739257812, 236.37298583984375, 238.91883850097656, 241.36561584472656, 243.72262573242188, 245.9986114501953, 248.2017822265625, 250.3395538330078, 252.41824340820312, 254.4441375732422, 256.4224853515625, 258.358154296875, 260.25604248046875, 262.11981201171875, 263.9532775878906, 265.7595520019531, 267.54180908203125, 269.30224609375, 271.0436706542969, 272.7677307128906, 274.47674560546875, 276.17181396484375, 277.8548583984375, 279.52691650390625, 281.18896484375, 282.84228515625, 284.48748779296875, 286.1255187988281, 287.7568054199219, 289.3819580078125, 291.0014343261719, 292.6157531738281, 294.2250061035156, 295.82958984375, 297.4297180175781, 299.0255126953125, 300.61712646484375, 302.2044982910156, 303.7882080078125, 305.36773681640625, 306.9434509277344, 308.51544189453125, 310.0833435058594, 311.6473388671875, 313.2076416015625, 314.76385498046875, 316.31622314453125, 317.86474609375, 319.40924072265625, 320.94970703125, 322.4862060546875, 324.0189208984375, 325.5475158691406, 327.072021484375, 328.5927429199219, 330.1095886230469, 331.62225341796875, 333.1312255859375, 334.6363830566406, 336.1376647949219, 337.63531494140625, 339.1293640136719, 340.61981201171875, 342.10687255859375, 343.5904846191406, 345.0709533691406, 346.54833984375, 348.0225830078125, 349.4942626953125, 350.96295166015625, 352.4293212890625, 353.8934020996094, 355.3551025390625, 356.81500244140625, 358.2731018066406, 359.72979736328125, 361.1847839355469, 362.63873291015625, 364.092041015625, 365.5445251464844, 366.9964599609375, 368.44818115234375, 369.9002685546875, 371.3524169921875, 372.80517578125, 374.25897216796875, 375.71380615234375, 377.170166015625, 378.6282043457031, 380.088134765625]
    plt.plot(x,y1,'bs',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 25, \beta_f/\beta_w=0.014$ (Platnium)')
    #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
    plt.plot(x2,y2,'-b',markersize=5,markeredgewidth=0.1,alpha=0.9)
           
    x = Psat[84:98]
    y1 = Tmin_exp[84:98]
    x2 = np.linspace(0, 3, num=100)
    y2 = [201.839111328125, 206.37599182128906, 210.636474609375, 214.64366149902344, 218.4195098876953, 221.98390197753906, 225.35543823242188, 228.5515899658203, 231.58786010742188, 234.47900390625, 237.2384490966797, 239.87828063964844, 242.40994262695312, 244.84368896484375, 247.1890869140625, 249.45448303222656, 251.64822387695312, 253.77688598632812, 255.84725952148438, 257.8653564453125, 259.836181640625, 261.76513671875, 263.6559753417969, 265.5129089355469, 267.3395690917969, 269.1390380859375, 270.9143371582031, 272.6676940917969, 274.4018859863281, 276.11834716796875, 277.81927490234375, 279.506103515625, 281.18048095703125, 282.84356689453125, 284.4962463378906, 286.1397705078125, 287.77447509765625, 289.40167236328125, 291.02191162109375, 292.635009765625, 294.242431640625, 295.84381103515625, 297.4398193359375, 299.03045654296875, 300.6162414550781, 302.19708251953125, 303.7733459472656, 305.3449401855469, 306.91192626953125, 308.4745788574219, 310.03271484375, 311.5865478515625, 313.1360168457031, 314.6807861328125, 316.22161865234375, 317.7576599121094, 319.2894287109375, 320.8164978027344, 322.3394775390625, 323.8580322265625, 325.37200927734375, 326.8813171386719, 328.38641357421875, 329.8870544433594, 331.38311767578125, 332.87493896484375, 334.36236572265625, 335.845458984375, 337.32421875, 338.798583984375, 340.26904296875, 341.7354431152344, 343.19775390625, 344.65618896484375, 346.1109313964844, 347.56207275390625, 349.00958251953125, 350.4537353515625, 351.89471435546875, 353.33245849609375, 354.7674255371094, 356.1998291015625, 357.6295166015625, 359.05670166015625, 360.48193359375, 361.9050598144531, 363.32647705078125, 364.746337890625, 366.1651611328125, 367.58282470703125, 368.9996643066406, 370.41595458984375, 371.83209228515625, 373.2481689453125, 374.6643981933594, 376.081298828125, 377.49884033203125, 378.917724609375, 380.33770751953125, 381.7596435546875]
    plt.plot(x,y1,'k^',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 41.67, \beta_f/\beta_w=0.014$ (Platnium)')
    #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
    plt.plot(x2,y2,'-k',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'ANN')
    
    #plt.ylim(200,700)
    plt.xlim(0,2)
    plt.xlabel(r'$P_{sat}$ [MPa]')
    plt.ylabel(r'$T_{min}$ [$\degree$C]')
    leg = plt.legend(loc='best',fancybox=False,numpoints=1)
    frame  = leg.get_frame()  
    frame.set_linewidth(0.5)
    plt.tight_layout()
    plt.show()
    fig.savefig('ANN_Tmin_vary_pressure_BfBw.pdf')
    plt.close() 
    
     
    ##########plot Tmin versus Psat for fixed Tsub = 0, vary Bw/Bf= 0.017, 0.053, 0.117 , vary L/D = 5.15 and 6.5##########
    fig=pylab.figure(figsize=(6,4))
    x11 = Psat[209:219]
    x22 = Psat[222:251]
    x = np.concatenate((x11,x22))
    y11 = Tmin_exp[209:219]
    y22 = Tmin_exp[222:251]
    y1 = np.concatenate((y11,y22))
    x2 = np.linspace(0, 3, num=100)
    y2 = [266.4869384765625, 293.2535095214844, 318.8741149902344, 343.3015441894531, 366.50341796875, 388.4625244140625, 409.17376708984375, 428.644287109375, 446.8906555175781, 463.93902587890625, 479.82177734375, 494.57781982421875, 508.2503662109375, 520.8865966796875, 532.5358276367188, 543.2491455078125, 553.0783081054688, 562.07568359375, 570.292724609375, 577.780517578125, 584.5882568359375, 590.764404296875, 596.3546142578125, 601.4037475585938, 605.953857421875, 610.045166015625, 613.715576171875, 617.0009765625, 619.9349365234375, 622.549560546875, 624.8740234375, 626.935791015625, 628.7611083984375, 630.373779296875, 631.7955322265625, 633.047119140625, 634.147705078125, 635.11474609375, 635.9644775390625, 636.7117919921875, 637.3701171875, 637.9522705078125, 638.469482421875, 638.9326171875, 639.3507080078125, 639.7327880859375, 640.0869140625, 640.4200439453125, 640.7388305664062, 641.0491943359375, 641.3564453125, 641.6651000976562, 641.9797973632812, 642.3043212890625, 642.6419677734375, 642.9959716796875, 643.3690185546875, 643.76318359375, 644.1806640625, 644.6234130859375, 645.0928955078125, 645.590576171875, 646.11767578125, 646.6746215820312, 647.2625732421875, 647.8824462890625, 648.5341796875, 649.2183837890625, 649.9354248046875, 650.685302734375, 651.468017578125, 652.28369140625, 653.1319580078125, 654.01318359375, 654.9268188476562, 655.8724365234375, 656.8499145507812, 657.8590087890625, 658.899169921875, 659.9698486328125, 661.0706787109375, 662.2015991210938, 663.3614501953125, 664.550048828125, 665.766845703125, 667.0117797851562, 668.283203125, 669.5816650390625, 670.9058837890625, 672.2557373046875, 673.6304321289062, 675.029541015625, 676.4525146484375, 677.8984985351562, 679.3673095703125, 680.8582763671875, 682.370849609375, 683.904296875, 685.458740234375, 687.03271484375]
    plt.plot(x,y1,'ro',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 6.5, \beta_f/\beta_w=0.017$ (Carbon steel)')
    #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
    plt.plot(x2,y2,'-r',markersize=5,markeredgewidth=0.1,alpha=0.9)
    
    x = Psat[194:208]
    y1 = Tmin_exp[194:208]
    x2 = np.linspace(0, 3, num=100)
    y2 = [284.5979919433594, 310.56744384765625, 335.3609924316406, 358.9423828125, 381.29071044921875, 402.3975830078125, 422.2667236328125, 440.91229248046875, 458.3570556640625, 474.63165283203125, 489.7724304199219, 503.821533203125, 516.8240966796875, 528.8281860351562, 539.8839111328125, 550.0426025390625, 559.35546875, 567.8740234375, 575.6484375, 582.7280883789062, 589.1614990234375, 594.9945068359375, 600.2715454101562, 605.0352172851562, 609.3260498046875, 613.182373046875, 616.64013671875, 619.7335815429688, 622.494873046875, 624.9539184570312, 627.138916015625, 629.075927734375, 630.7890625, 632.301513671875, 633.634033203125, 634.8060302734375, 635.8355102539062, 636.739013671875, 637.5322265625, 638.2288818359375, 638.8419799804688, 639.38330078125, 639.8638916015625, 640.2935791015625, 640.6815185546875, 641.0360107421875, 641.3646240234375, 641.673828125, 641.9703369140625, 642.2591552734375, 642.546142578125, 642.835205078125, 643.1304931640625, 643.4359130859375, 643.754638671875, 644.0896606445312, 644.443115234375, 644.81787109375, 645.215576171875, 645.637939453125, 646.086669921875, 646.5626220703125, 647.067626953125, 647.6019287109375, 648.1663208007812, 648.7615966796875, 649.3883666992188, 650.046875, 650.7371826171875, 651.4597778320312, 652.2144165039062, 653.0010375976562, 653.8200073242188, 654.6708374023438, 655.5530395507812, 656.4669799804688, 657.4119873046875, 658.3878173828125, 659.3939208984375, 660.4302368164062, 661.4960327148438, 662.591064453125, 663.7147216796875, 664.8663940429688, 666.0458374023438, 667.2525024414062, 668.48583984375, 669.7451782226562, 671.030029296875, 672.3402099609375, 673.6748046875, 675.0330810546875, 676.415283203125, 677.820068359375, 679.24755859375, 680.696533203125, 682.1668701171875, 683.658203125, 685.16943359375, 686.7009887695312]
    plt.plot(x,y1,'bs',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 6.5, \beta_f/\beta_w=0.053$ (Stainless steel)')
    #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
    plt.plot(x2,y2,'-b',markersize=5,markeredgewidth=0.1,alpha=0.9)
    
    x = Psat[166:193]
    y1 = Tmin_exp[166:193]
    x2 = np.linspace(0, 3, num=100)
    y2 = [315.0618896484375, 339.47808837890625, 362.6962890625, 384.699462890625, 405.4822082519531, 425.0506896972656, 443.4200439453125, 460.614501953125, 476.6645202636719, 491.6068420410156, 505.48223876953125, 518.335205078125, 530.212646484375, 541.1632690429688, 551.2364501953125, 560.4818115234375, 568.9490966796875, 576.6866455078125, 583.74267578125, 590.1629028320312, 595.992431640625, 601.2738647460938, 606.04833984375, 610.354736328125, 614.2306518554688, 617.7105712890625, 620.827880859375, 623.6134033203125, 626.0966796875, 628.3048095703125, 630.2633056640625, 631.9959716796875, 633.525146484375, 634.8712768554688, 636.0537109375, 637.0895385742188, 637.9957275390625, 638.7874755859375, 639.478271484375, 640.0811767578125, 640.6080322265625, 641.0697021484375, 641.4761352539062, 641.8365478515625, 642.1592407226562, 642.4514770507812, 642.7205810546875, 642.972900390625, 643.2137451171875, 643.4486083984375, 643.68212890625, 643.9183959960938, 644.1614990234375, 644.41455078125, 644.6807861328125, 644.962646484375, 645.2626953125, 645.5833129882812, 645.926025390625, 646.2926025390625, 646.6843872070312, 647.1027221679688, 647.5486450195312, 648.0228271484375, 648.5263671875, 649.0594482421875, 649.622802734375, 650.2167358398438, 650.84130859375, 651.496826171875, 652.1835327148438, 652.9013061523438, 653.6500244140625, 654.429443359375, 655.239990234375, 656.08056640625, 656.9515380859375, 657.8524780273438, 658.7828979492188, 659.7425537109375, 660.7308349609375, 661.7479248046875, 662.79248046875, 663.864990234375, 664.9642333984375, 666.090087890625, 667.2421264648438, 668.4198608398438, 669.6228637695312, 670.85009765625, 672.1016845703125, 673.376708984375, 674.6751708984375, 675.9962158203125, 677.3389892578125, 678.703857421875, 680.0896606445312, 681.49609375, 682.9227905273438, 684.3692626953125]
    plt.plot(x,y1,'gv',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 6.5, \beta_f/\beta_w=0.117$ (Zirconium)')
    #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
    plt.plot(x2,y2,'-g',markersize=5,markeredgewidth=0.1,alpha=0.9)
    
    x = Psat[19:35]
    y1 = Tmin_exp[19:35]
    x2 = np.linspace(0, 3, num=100)
    y2 = [273.1280517578125, 300.3673095703125, 326.5037536621094, 351.4905700683594, 375.29498291015625, 397.8983154296875, 419.2935791015625, 439.48504638671875, 458.4864501953125, 476.31982421875, 493.0143737792969, 508.6046447753906, 523.1304931640625, 536.6343994140625, 549.1622314453125, 560.7611083984375, 571.4794921875, 581.3660888671875, 590.468994140625, 598.83642578125, 606.514892578125, 613.5503540039062, 619.986083984375, 625.8648681640625, 631.2266235351562, 636.1103515625, 640.552490234375, 644.5872802734375, 648.247802734375, 651.5645751953125, 654.5663452148438, 657.280029296875, 659.7313842773438, 661.943603515625, 663.9381103515625, 665.736328125, 667.356689453125, 668.816650390625, 670.1329956054688, 671.320068359375, 672.3925170898438, 673.36279296875, 674.2427978515625, 675.04345703125, 675.77490234375, 676.446044921875, 677.0655517578125, 677.6412353515625, 678.18017578125, 678.6883544921875, 679.1724853515625, 679.6372680664062, 680.0879516601562, 680.5286865234375, 680.9636840820312, 681.3966064453125, 681.8308715820312, 682.2691040039062, 682.71435546875, 683.168701171875, 683.6341552734375, 684.1131591796875, 684.60693359375, 685.1170043945312, 685.6453857421875, 686.1920166015625, 686.7587890625, 687.3463134765625, 687.955322265625, 688.5862426757812, 689.2401123046875, 689.916748046875, 690.6168823242188, 691.3404541015625, 692.0880126953125, 692.8594970703125, 693.65478515625, 694.4742431640625, 695.317626953125, 696.184814453125, 697.075927734375, 697.9905395507812, 698.9287109375, 699.8900146484375, 700.8740234375, 701.8812255859375, 702.9104614257812, 703.9622802734375, 705.035888671875, 706.1307373046875, 707.2469482421875, 708.3843383789062, 709.5419921875, 710.7198486328125, 711.91748046875, 713.1348876953125, 714.371337890625, 715.62646484375, 716.9000244140625, 718.1917724609375]
    plt.plot(x,y1,'k^',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'$L/D = 5.15, \beta_f/\beta_w=0.117$ (Zirconium)')
    #plt.errorbar(x,y1,yerr=0.2*y1,fmt='',linestyle="None",color='k')
    plt.plot(x2,y2,'-k',markersize=5,markeredgewidth=0.1,alpha=0.9,label=r'ANN')
    
    #plt.ylim(200,700)
    plt.xlim(0,3)
    plt.xlabel(r'$P_{sat}$ [MPa]')
    plt.ylabel(r'$T_{min}$ [$\degree$C]')
    leg = plt.legend(loc='best',fancybox=False,numpoints=1)
    frame  = leg.get_frame()  
    frame.set_linewidth(0.5)
    plt.tight_layout()
    plt.show()
    fig.savefig('ANN_Tmin_vary_pressure_LD.pdf')
    plt.close()    
    
if __name__ == '__main__':
    
    Calculate()
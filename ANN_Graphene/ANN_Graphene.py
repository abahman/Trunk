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
    visc_exp = float(data[i][0])
    Temp = float(data[i][1])
    graphene_vol = float(data[i][2])
    surfactant_rho = float(data[i][3])
    surfactant_vol = float(data[i][4])
#     surfactant_typ = str(data[i][5])
    i=i+1
    
    while i < (end - start+1):
        visc_exp = np.append(visc_exp,float(data[i][0]))
        Temp = np.append(Temp,float(data[i][1]))
        graphene_vol = np.append(graphene_vol,float(data[i][2])) 
        surfactant_rho = np.append(surfactant_rho,float(data[i][3]))
        surfactant_vol = np.append(surfactant_vol,float(data[i][4]))
#         surfactant_typ = np.append(surfactant_typ,str(data[i][5]))
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
    end=88
    filename = 'Data_Collection_new.csv'
    [visc_exp,Temp,graphene_vol,surfactant_rho,surfactant_vol] = Import(start,end,filename)
    

    mode = 'training'
    
    from keras.models import Model,Sequential,load_model
    from keras.layers import Input,Flatten,Dense,LSTM,merge,Dropout,concatenate
    #from keras.engine import merge # from Keras version 1.2.2
    #from keras.layers.merge import concatenate
    from keras.utils.vis_utils import plot_model
    from keras.callbacks import TensorBoard
    from keras import regularizers
    

    #Define inputs
    Temp = Temp.reshape(-1,1,1) 
    graphene_vol = graphene_vol.reshape(-1,1,1) 
    surfactant_rho = surfactant_rho.reshape(-1,1,1)
    surfactant_vol = surfactant_vol.reshape(-1,1,1)
    
    
    #Normalize all parameters
    visc_exp_norm = Normalize(visc_exp, 0.452954654, 1.269923504)
    Temp_norm = Normalize(Temp, 20, 50)
    graphene_vol_norm = Normalize(graphene_vol, 0.0, 0.5303025)
    surfactant_rho_norm = Normalize(surfactant_rho, 0.0, 1.4)
    surfactant_vol_norm = Normalize(surfactant_vol, 0.0, 0.79545375)
    
    if mode == 'training':
        visible1 = Input(shape=(1,1), name='Temp')
        visible2 = Input(shape=(1,1), name='graphene_vol') 
        visible3 = Input(shape=(1,1), name='surfactant_rho') 
        visible4 = Input(shape=(1,1), name='surfactant_vol')
    
        shared_lstm = LSTM(4)
    
        encoded_a = shared_lstm(visible1)
        encoded_b = shared_lstm(visible2) 
        encoded_c = shared_lstm(visible3) 
        encoded_d = shared_lstm(visible4)

    
        #Merge inputs
        #merged = merge([encoded_a,encoded_b,encoded_c,encoded_d,encoded_e,encoded_f],mode='concat',concat_axis=-1) #deprecated
        merged = concatenate([encoded_a,encoded_b,encoded_c,encoded_d],axis=-1)
        
        #interpretation model
        hidden2 = Dense(12,activation='tanh')(merged) #hidden1 = Dense(256, activation='tanh')(merged) ###'relu' shows good results
#         hidden2 = Dense(12, activation = 'tanh')(hidden1)
        #hidden3 = Dropout(0.2, noise_shape=None, seed=None)(hidden2)
        #hidden3 = Dense(100, activation = 'tanh')(hidden2)
        #hidden4 = Dense(32, activation = 'tanh')(hidden3)
        
        output1 = Dense(1, activation = 'linear',name='visc')(hidden2)

       
        model = Model([visible1,visible2,visible3,visible4],
                        [output1])
        
#         plot_model(model, to_file='model.pdf',show_shapes=True,show_layer_names=True)
        
        model.compile(optimizer='adamax',loss='mse',metrics=['mae']) #model.compile(optimizer='adamax',loss=['mse','mse','mse','mse','mse','mse']) #metrics are not included in the training
        
        history = model.fit([Temp_norm,graphene_vol_norm,surfactant_rho_norm,surfactant_vol_norm],
                            [visc_exp_norm],
                            epochs=1600,
                            batch_size=10, #increase the batch size results in faster compiler and high error, while smaller batch size results in slower compiler and slightly accurate model
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
        fig.savefig('ANN_history_visc.pdf')
        
        # Save the model
        model.save('ANN_model_visc.h5')
    
    elif mode == 'run':
    
        # Load the model
        model = load_model('ANN_model_visc.h5')
    
    # Run the model
    visc_ANN = model.predict([Temp_norm,graphene_vol_norm,surfactant_rho_norm,surfactant_vol_norm])
    visc_ANN = DeNormalize(visc_ANN.reshape(-1), 0.452954654, 1.269923504)
    
    # evaluate the model
    scores = model.evaluate([Temp_norm,graphene_vol_norm,surfactant_rho_norm,surfactant_vol_norm],[visc_exp_norm])
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
#     # extract the weight and bias
#     weights = model.layers[0].get_weights()[0]
#     biases = model.layers[0].get_weights()[1]
    

    # Save the architecture of a model, and not its weights or its training configuration
    # save as JSON
    # json_string = model.to_json()
    
    # save as YAML
    # yaml_string = model.to_yaml()


    for i in range(0,(end-start+1)):


        data_calc = {'visc':[visc_ANN[i]]}  #data_calc = {'Tdis':[T[i]],'mdot':[Mref[i]],'mdot_inj':[Minj[i]], 'Wdot':[W[i]],'etaoa':[eta_s[i]],'fq':[Q[i]/W[i]]} 
            
        
        # Write to Excel
        filename = os.path.dirname(__file__)+'/visc_output.xlsx'
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
    n_len = len(visc_ANN)
    n_split = int(np.floor(sep_val*n_len))
    n_training = int(n_len-n_split-1)


    # Validation Tmin
    fig=pylab.figure(figsize=(4,4))

    plt.plot(visc_ANN[:n_training],visc_exp[:n_training],'ro',ms = 3,mec='black',mew=0.5,label='Training points')
    plt.plot(visc_ANN[-n_split:],visc_exp[-n_split:],'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
    plt.text(1,0.5,'R$^2$ = {:0.01f}%\n'.format(Rsquared(visc_exp,visc_ANN)*100)+'MAE = {:0.01f}%\n'.format(mape(visc_ANN,visc_exp))+'RMSE = {:0.01f}%'.format(rmse(visc_ANN,visc_exp)),ha='left',va='center',fontsize = 8)

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
    plt.legend(loc=2,fontsize=9)
    plt.tight_layout(pad=0.2)        
    plt.tick_params(direction='in')
    plt.show()
    fig.savefig('ANN_visc.pdf')
    
    print ('visc:',REmean(visc_exp,visc_ANN),Rsquared(visc_exp,visc_ANN)*100)


    
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
    
    
    
    
    
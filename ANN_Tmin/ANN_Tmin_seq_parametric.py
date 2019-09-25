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
        i=i+1
        Data = [Tmin,Tsub,Psat,Mat,LD,Bf,Bw,BfBw]
    
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
    
    
    from keras.models import Model,Sequential,load_model
    from keras.layers import Input,Flatten,Dense,LSTM,merge,Dropout,concatenate
    #from keras.engine import merge # from Keras version 1.2.2
    #from keras.layers.merge import concatenate
    from keras.layers import GaussianNoise
    from keras.utils import plot_model
    from keras.callbacks import TensorBoard
    from keras import regularizers
    
    SC = np.array([]) #empty score array
    
    s = np.linspace(0, 3, num=100)
    for i in range(len(s)):
        Tmin_exp = 0
        Tsub = 0
        Psat = s[i]
        LD = 24.4
        BfBw = 0.014
        
        #Normalize all parameters
        Tsub_norm = Normalize(Tsub, 0, 39.84150546)
        Psat_norm = Normalize(Psat, 0.001185867, 3.003378378)
        LD_norm = Normalize(LD, 2.67, 63.5)
        BfBw_norm = Normalize(BfBw, 0.001989845, 0.530923555)
        
        #convert to numpy array
        Tsub_norm = np.array(Tsub_norm)
        Psat_norm = np.array(Psat_norm)
        LD_norm = np.array(LD_norm)
        BfBw_norm = np.array(BfBw_norm)
        
        # split into input (X) and output (Y) variables
        X = np.column_stack((Tsub_norm, Psat_norm))
        X = np.column_stack((X, LD_norm))
        X = np.column_stack((X, BfBw_norm))
    #     Y = Tmin_exp_norm
    
            
        # Load the model
        model = load_model('ANN_model_Tmin.h5',custom_objects={'coeff_determination': coeff_determination})
                
            
        # Run the model
        Tmin_ANN = model.predict(X)
        Tmin_ANN = DeNormalize(Tmin_ANN.reshape(-1), 206.8841, 727.8873239) #W = DeNormalize(W.reshape(-1),1000,8000)
        
        SC = np.append(SC,Tmin_ANN)
    
    for i in range(len(SC)):
        print(str(SC[i])+str(",")),
        

if __name__ == '__main__':
    
    Calculate()
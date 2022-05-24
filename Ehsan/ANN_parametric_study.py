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
        ref = 0
        oil = 0
        angel = 0
        T_sat = 0
        T_sup = 0
        D = 0
        m_r = 0
        m_o = 0
        OCR= s[i]
        
        #Normalize all parameters
        ref_norm = Normalize(ref, 0, 1)
        oil_norm = Normalize(oil, 0, 1)
        angel_norm = Normalize(angel, 0.0, 1)
        T_sat_norm = Normalize(T_sat, 0.0, 1)
        T_sup_norm = Normalize(T_sup, 0.0, 1)
        D_norm = Normalize(D, 0.0, 1)
        m_r_norm = Normalize(m_r, 0.0, 1)
        m_o_norm = Normalize(m_o, 0.0, 1)
        OCR_norm = Normalize(OCR, 0.0, 1)
        #convert to numpy array
        ref_norm = np.array(ref_norm)
        oil_norm = np.array(oil_norm)
        angel_norm = np.array(angel_norm)
        T_sat_norm = np.array(T_sat_norm)
        T_sup_norm = np.array(T_sup_norm)
        D_norm = np.array(D_norm)
        m_r_norm = np.array(m_r_norm)
        m_o_norm = np.array(m_o_norm)
        OCR_norm = np.array(OCR_norm)
        # split into input (X) and output (Y) variables
        X = np.column_stack((ref_norm, oil_norm))
        X = np.column_stack((X, angel_norm))
        X = np.column_stack((X, T_sat_norm))
        X = np.column_stack((X, T_sup_norm))
        X = np.column_stack((X, D_norm))
        X = np.column_stack((X, m_r_norm))
        X = np.column_stack((X, m_o_norm))
        X = np.column_stack((X, OCR_norm))
           
            
        # Load the model
        model = load_model('ANN_model.h5',custom_objects={'coeff_determination': coeff_determination})
                
            
        # Run the model
        OR_ANN = model.predict(X)
        OR_ANN = DeNormalize(OR_ANN.reshape(-1), 0.0, 1)
        
        SC = np.append(SC,OR_ANN)
    
    for i in range(len(SC)):
        print(str(SC[i])+str(",")),
        

if __name__ == '__main__':
    
    Calculate()
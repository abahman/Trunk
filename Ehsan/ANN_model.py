import os,sys
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import optimize,stats
import pandas as pd
from openpyxl import load_workbook
from keras import backend as K
import DataIO
import pickle
import time

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
    ID = float(data[i][0])
    P0 = float(data[i][1])
    P1 = float(data[i][2])
    P2 = float(data[i][3])
    P3 = float(data[i][4])
    P4 = float(data[i][5])
    P5 = float(data[i][6])
    P6 = float(data[i][7])
    v1_exp = float(data[i][8])
    v2_exp = float(data[i][9])
    v3_exp = float(data[i][10])
    
    i=i+1
    
    while i < (end - start+1):
        ID = np.append(ID,float(data[i][0]))
        P0 = np.append(P0,float(data[i][1]))
        P1 = np.append(P1,float(data[i][2]))
        P2 = np.append(P2,float(data[i][3]))
        P3 = np.append(P3,float(data[i][4]))
        P4 = np.append(P4,float(data[i][5]))
        P5 = np.append(P5,float(data[i][6]))
        P6 = np.append(P6,float(data[i][7]))
        v1_exp = np.append(v1_exp,float(data[i][8]))
        v2_exp = np.append(v2_exp,float(data[i][9]))
        v3_exp = np.append(v3_exp,float(data[i][10])) 

        i=i+1
        Data = [ID,P0,P1,P2,P3,P4,P5,P6,v1_exp,v2_exp,v3_exp]  

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
    end=25600
    filename = 'Data_file.csv'
    
    #Define inputs
    [ID,P0,P1,P2,P3,P4,P5,P6,v1_exp,v2_exp,v3_exp] = Import(start,end,filename)
    
    mode = 'training'
    
    from keras.models import Model,Sequential,load_model
    from keras.layers import Input,Flatten,Dense,LSTM,merge,Dropout,concatenate
    from keras.layers import GaussianNoise
    from keras.utils.vis_utils import plot_model
    from keras.callbacks import TensorBoard
    from keras import regularizers
    
    
    #Normalize all parameters + convert to numpy array
    P0_norm = np.array(Normalize(P0, 22, 26))
    P1_norm = np.array(Normalize(P1, 0.05, 0.2))
    P2_norm = np.array(Normalize(P2, 0.05, 0.14))
    P3_norm = np.array(Normalize(P3, 2, 14))
    P4_norm = np.array(Normalize(P4, 0.6, 0.9))
    P5_norm = np.array(Normalize(P5, 25, 28))
    P6_norm = np.array(Normalize(P6, 13, 28))
    v1_exp_norm = np.array(Normalize(v1_exp, 5201.68, 12907.64))
    v2_exp_norm = np.array(Normalize(v2_exp, 0, 776.5))
    v3_exp_norm = np.array(Normalize(v3_exp, 1629.05, 4812.98))    
    
    # split into input (X) and output (Y) variables
    X = np.column_stack((P0_norm, P1_norm))
    X = np.column_stack((X, P2_norm))
    X = np.column_stack((X, P3_norm))
    X = np.column_stack((X, P4_norm))
    X = np.column_stack((X, P5_norm))
    X = np.column_stack((X, P6_norm))
    Y = np.column_stack((v1_exp_norm, v2_exp_norm))
    Y = np.column_stack((Y, v3_exp_norm))
    
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
            model.add(Dense(i+30, input_dim=7, activation='tanh')) #init='uniform' #use_bias = True, bias_initializer='zero' #4 is perfect
            #model.add(GaussianNoise(0.1))
            #model.add(Dropout(0.2)) #Dropout is a technique where randomly selected neurons are ignored during training.
            #model.add(Dense(i+30, activation='tanh'))
            #model.add(GaussianNoise(0.1))
            #model.add(Dense(i+12, activation='tanh'))
            model.add(Dense(3, activation='linear'))
              
            #plot_model(model, to_file='model.pdf',show_shapes=True,show_layer_names=True)
      
            # Compile model
            model.compile(optimizer='Adam',loss='mse',metrics=['mae',coeff_determination])
              
            # fit the model
            history = model.fit(X_train,
                                Y_train,
                                epochs=30 , #Cut the epochs in half when using sequential 
                                batch_size=500, #increase the batch size results in faster compiler and high error, while smaller batch size results in slower compiler and slightly accurate model
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
            # plt.show()     
            fig.savefig('ANN_history_loss.pdf')
    
        #   #History plot for accuracy
            fig=pylab.figure(figsize=(6,4))
            plt.semilogy(history.history['coeff_determination'])
            plt.semilogy(history.history['val_coeff_determination'])
            plt.ylabel('R$^2$')
            plt.xlabel('epochs')
            plt.legend(['Train', 'Test'], loc='upper right',fontsize=12)
            plt.ylim(0.001,1)
            plt.tight_layout(pad=0.2)  
            plt.tick_params(direction='in')
            # plt.show()     
            fig.savefig('ANN_history_acc.pdf')
                    
            # Save the model
            model.save('ANN_model.h5')
            # Save the history
            H = history.history
            with open('HistoryDict.pickle', 'wb') as handle:
                pickle.dump(H, handle)
                
        elif mode == 'run':
        
            # Load the model
            model = load_model('ANN_model.h5',custom_objects={'coeff_determination': coeff_determination})
            # Load the history
            with open('HistoryDict.pickle', 'rb') as handle:
                H = pickle.load(handle)
  
        # Run the model
        V1_ANN = model.predict(X)
        
        from sklearn.metrics import mean_squared_error
        Y_pred = model.predict(X)
        
        Y_test = Y
        print("y1 MSE:%.4f" % mean_squared_error(Y_test[:,0], Y_pred[:,0]))
        print("y2 MSE:%.4f" % mean_squared_error(Y_test[:,1], Y_pred[:,1]))
        print("y3 MSE:%.4f" % mean_squared_error(Y_test[:,2], Y_pred[:,2]))
        
        x_ax = range(len(X))
        
        fig=pylab.figure(figsize=(6,4))
        plt.scatter(x_ax, Y_test[:,0],  s=6, label="y1-test")
        plt.plot(x_ax, Y_pred[:,0], label="y1-pred")
        plt.scatter(x_ax, Y_test[:,1],  s=6, c = 'g', marker='^', label="y2-test")
        plt.plot(x_ax, Y_pred[:,1], label="y2-pred")
        plt.scatter(x_ax, Y_test[:,2],  s=6, c = 'r', marker='s', label="y3-test")
        plt.plot(x_ax, Y_pred[:,2], label="y3-pred")
        plt.legend()
        plt.show()
        fig.savefig('new.pdf')
        
        
        
    #     V1 = V_ANN[:,0]
    #     V2 = V_ANN[:,1]
    #     V3 = V_ANN[:,2]
    #     print (V_ANN)
    #     print (V1)
    #     print (V2)
    #     print (V3)
    #     #V_ANN = DeNormalize(V_ANN.reshape(-1), 0.77, 20.33)
    #
    #     V1_ANN = DeNormalize(V1, 5201.68, 12907.64)
    #     V2_ANN = DeNormalize(V2, 0, 776.5)
    #     V3_ANN = DeNormalize(V3, 1629.05, 4812.98)
    #
    #     # evaluate the model (for the last batch)
    #     scores = model.evaluate(X,Y)
    #     SC = np.append(SC,scores[1]*100)
    #     ms = np.append(ms,mse(V1_ANN, v1_exp))
    #     ma = np.append(ma,mape(V1_ANN, v1_exp))
    #     R2 = np.append(R2,Rsquared(v1_exp, V1_ANN))
    #
    #     ms = np.append(ms,mse(V2_ANN, v2_exp))
    #     ma = np.append(ma,mape(V2_ANN, v2_exp))
    #     R2 = np.append(R2,Rsquared(v2_exp, V2_ANN))
    #
    #     ms = np.append(ms,mse(V3_ANN, v3_exp))
    #     ma = np.append(ma,mape(V3_ANN, v3_exp))
    #     R2 = np.append(R2,Rsquared(v3_exp, V3_ANN))
    #
    #     print('')
    #     print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    #     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    #     print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    #
    #     # extract the weight and bias
    #     weights = model.layers[0].get_weights()[0]
    #     biases = model.layers[0].get_weights()[1]
    #
    #     #to change the precision of printed numbers
    #     np.set_printoptions(precision=4, suppress=True,
    #                    threshold=10000,
    #                    linewidth=150)
    #     print('')
    #     print ('weights = ', weights.transpose())
    #     print ('biases = ', biases)
    #
    #     # Save the architecture of a model, and not its weights or its training configuration
    #     # save as JSON
    #     # json_string = model.to_json()
    #
    #     # save as YAML
    #     # yaml_string = model.to_yaml()
    # print('')
    # for i in range(len(SC)):
    #     print (SC[i])
    #     print (mape(V1_ANN[i], v1_exp[i]))
    #     print (mape(V2_ANN[i], v2_exp[i]))
    #     print (mape(V3_ANN[i], v3_exp[i]))
    #     print (ms[i],ma[i],R2[i])    
        
    # to SAVE into excel file
    # for i in range(0,(end-start+1)):
    #
    #     data_calc = {'OR':[OR_ANN[i]]}
    #
    #
    #     # Write to Excel
    #     filename = os.path.dirname(__file__)+'/OR_output.xlsx'
    #     xl = pd.read_excel(filename, sheet_name='ANN_Validation')
    #
    #     df = pd.DataFrame(data=data_calc)
    #
    #     df.reindex(columns=xl.columns)
    #     df_final=xl.append(df,ignore_index=True)
    #     df_final.tail()
    #
    #     book = load_workbook(filename)
    #     writer = pd.ExcelWriter(filename, engine='openpyxl',index=False)
    #     writer.book = book
    #     writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    #     df_final.to_excel(writer,index=False,sheet_name='ANN_Validation')
    #
    #     # 
    #     writer.save()

    V1_ANN = Y_pred[:,0]
    Y = Y[:,0]    
    #Separate testing from calibrating
    sep_val = 0.3
    n_len = len(V1_ANN)
    n_split = int(np.floor(sep_val*n_len))
    n_training = int(n_len-n_split-1)


#     # Validation OR
    fig=pylab.figure(figsize=(4,4))
 
    plt.plot(V1_ANN[:n_training],Y[:n_training],'ro',ms = 3,mec='black',mew=0.5,label='Training points')
    plt.plot(V1_ANN[-n_split:],Y[-n_split:],'b*',ms = 4,mec='black',mew=0.5,label='Testing points')
    plt.text(0.6,0.2,'R$^2$ = {:0.03f}\n'.format(Rsquared(Y,V1_ANN))+'MAE = {:0.01f}%\n'.format(mape(V1_ANN,Y))+'RMSE = {:0.01f}%\n'.format(rmse(V1_ANN,Y)),ha='left',va='center',fontsize = 8)
 
    plt.xlabel('Y1$_{pred}$ [-]')
    plt.ylabel('Y1$_{exp}$ [-]')
 
    Tmin = 0
    Tmax = 1
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
    fig.savefig('ANN_Y1.pdf')

    # print ('Method:','MSE','MAPE','Rsquared')
    # print ('V1_ANN:',mse(V1_ANN,v1_exp),mape(V1_ANN, v1_exp),Rsquared(v1_exp,V1_ANN))
    # print ('V2_ANN:',mse(V2_ANN,v2_exp),mape(V2_ANN, v3_exp),Rsquared(v2_exp,V2_ANN))
    # print ('V3_ANN:',mse(V3_ANN,v3_exp),mape(V3_ANN, v3_exp),Rsquared(v3_exp,V3_ANN))

    
#     for i in range(len(Tmin_ANN)): # print the measure absolute error (%) for all data
#         print (REmean(OR_exp[i],OR_ANN[i])*100)
    #error = np.array([0,1,2,3,4,5,6,7,8,9,10,12,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
    #set = np.array([0,14.51187335,27.70448549,41.16094987,50.92348285,61.7414248,72.55936675,81.26649077,86.80738786,89.97361478,93.13984169,95.77836412,97.88918206,99.73614776,99.73614776,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100])

    #Validation with literatures data
    #"Import Experimental Data"
    # start=1
    # end=200
    # filename = 'Data_Literature.csv'
    # #Define inputs
    # [ref,oil,angel,T_sat,T_sup,D,m_r,m_o,OCR,OR_source1] = Import(start,end,filename)
    # #Normalize all parameters
    # ref_norm = Normalize(ref, 0, 1)
    # oil_norm = Normalize(oil, 0, 1)
    # angel_norm = Normalize(angel, 0.0, 1)
    # T_sat_norm = Normalize(T_sat, 0.0, 1)
    # T_sup_norm = Normalize(T_sup, 0.0, 1)
    # D_norm = Normalize(D, 0.0, 1)
    # m_r_norm = Normalize(m_r, 0.0, 1)
    # m_o_norm = Normalize(m_o, 0.0, 1)
    # OCR_norm = Normalize(OCR, 0.0, 1)
    # OR_source1_norm = Normalize(OR_source1, 0.0, 1)
    # #convert to numpy array
    # ref_norm = np.array(ref_norm)
    # oil_norm = np.array(oil_norm)
    # angel_norm = np.array(angel_norm)
    # T_sat_norm = np.array(T_sat_norm)
    # T_sup_norm = np.array(T_sup_norm)
    # D_norm = np.array(D_norm)
    # m_r_norm = np.array(m_r_norm)
    # m_o_norm = np.array(m_o_norm)
    # OCR_norm = np.array(OCR_norm)
    # OR_source1_norm = np.array(OR_source1_norm)
    # # split into input (X) and output (Y) variables
    # X = np.column_stack((ref_norm, oil_norm))
    # X = np.column_stack((X, angel_norm))
    # X = np.column_stack((X, T_sat_norm))
    # X = np.column_stack((X, T_sup_norm))
    # X = np.column_stack((X, D_norm))
    # X = np.column_stack((X, m_r_norm))
    # X = np.column_stack((X, m_o_norm))
    # X = np.column_stack((X, OCR_norm))
    # Y = OR_source1_norm
    # # Load the model
    # model = load_model('ANN_model_Tmin.h5')
    
    # Run the model
    #Validation
    OR_ANN_valid = model.predict(X_valid)
    OR_ANN_valid = DeNormalize(OR_ANN_valid.reshape(-1),  0.77, 20.33)
    OR_exp_valid = DeNormalize(Y_valid.reshape(-1),  0.77, 20.33) 
    #Training
    OR_ANN_train = model.predict(X_train)
    OR_ANN_train = DeNormalize(OR_ANN_train.reshape(-1), 0.77, 20.33)
    OR_exp_train = DeNormalize(Y_train.reshape(-1), 0.77, 20.33) 
    #Testing
    OR_ANN_test = model.predict(X_test)
    OR_ANN_test = DeNormalize(OR_ANN_test.reshape(-1), 0.77, 20.33)
    OR_exp_test = DeNormalize(Y_test.reshape(-1), 0.77, 20.33) 
        
    # plot all data
    fig=pylab.figure(figsize=(4,4))
    plt.plot(OR_ANN_train,OR_exp_train,'ro',ms = 4,mec='black',mew=0.5,label='Training points')
    plt.plot(OR_ANN_test,OR_exp_test,'b*',ms = 6,mec='black',mew=0.5,label='Testing points')
    plt.plot(OR_ANN_valid,OR_exp_valid,'g^',ms = 6,mec='black',mew=0.5,label='Validation points')
    plt.text(12,4.5,'R$^2$ = {:0.03f}\n'.format(Rsquared(OR_exp,OR_ANN))+'MAE = {:0.02f}%\n'.format(mape(OR_ANN,OR_exp))+'MSE = {:0.03f}\n'.format(mse(OR_ANN,OR_exp)),ha='left',va='center',fontsize = 10)
    plt.xlabel('OR$_{pred}$ [g/m]')
    plt.ylabel('OR$_{exp}$ [g/m]')
    Vmin = 0
    Vmax = 22
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
    fig.savefig('ANN_all.pdf')
    plt.close()
    
    # plots (training)
    fig=pylab.figure(figsize=(4,4))
    plt.plot(OR_ANN_train,OR_exp_train,'ro',ms = 4,mec='black',mew=0.5,label='Training points')
    plt.text(12,4.5,'R$^2$ = {:0.03f}\n'.format(Rsquared(OR_exp_train,OR_ANN_train))+'MAE = {:0.02f}%\n'.format(mape(OR_ANN_train,OR_exp_train))+'MSE = {:0.03f}\n'.format(mse(OR_ANN_train,OR_exp_train)),ha='left',va='center',fontsize = 10)
    plt.xlabel('OR$_{pred}$ [g/m]')
    plt.ylabel('OR$_{exp}$ [g/m]')
    Tmin = 0
    Tmax = 22
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
    fig.savefig('ANN_training.pdf')  
    plt.close()
     
    #  plots (testing)
    fig=pylab.figure(figsize=(4,4))
    plt.plot(OR_ANN_test,OR_exp_test,'b*',ms = 6,mec='black',mew=0.5,label='Testing points')
    plt.text(12,4.5,'R$^2$ = {:0.03f}\n'.format(Rsquared(OR_exp_test,OR_ANN_test))+'MAE = {:0.02f}%\n'.format(mape(OR_ANN_test,OR_exp_test))+'MSE = {:0.03f}\n'.format(mse(OR_ANN_test,OR_exp_test)),ha='left',va='center',fontsize = 10)
    plt.xlabel('OR$_{pred}$ [g/m]')
    plt.ylabel('OR$_{exp}$ [g/m]')
    Tmin = 0
    Tmax = 22
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
    fig.savefig('ANN_testing.pdf')
    plt.close()
     
    # plots (validation)
    fig=pylab.figure(figsize=(4,4))
    plt.plot(OR_ANN_valid,OR_exp_valid,'g^',ms = 6,mec='black',mew=0.5,label='Validation points')
    plt.text(12,4.5,'R$^2$ = {:0.03f}\n'.format(Rsquared(OR_exp_valid,OR_ANN_valid))+'MAE = {:0.02f}%\n'.format(mape(OR_ANN_valid,OR_exp_valid))+'MSE = {:0.03f}\n'.format(mse(OR_ANN_valid,OR_exp_valid)),ha='left',va='center',fontsize = 10)
    plt.xlabel('OR$_{pred}$ [g/m]')
    plt.ylabel('OR$_{exp}$ [g/m]')
    Tmin = 0
    Tmax = 22
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
    fig.savefig('ANN_validation.pdf') 
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
#     fig.savefig('ANN_distribution.pdf') 
#     plt.close()
#     
#     ##########parametric study##########
 
 
 
    
if __name__ == '__main__':
    start_time = time.time()
    Calculate()
    print("--- %s seconds ---" % (time.time() - start_time))
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:12:47 2017

@author: orkan
"""
# Windows -> anaconda -> command prompt
#  pip install coolprop
#
from pprint import pprint
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

def F2K(x):
    return (x-32.)/1.8 + 273.15

def K2F(x):
    return (x - 273.15) * 1.8 + 32.

def Psi2Pa(x):
    return x*6894.76


def load_data():

    #Reading Data from Excel Sheet
    data = pd.read_excel(
        "./Processed_Data.xlsx",
        sheetname="Summary",
        header=0,
        skiprows=[1,1]
    )

    return data

def convert_units(data):

    temp_cols = data.filter(regex="TC[0-9]*", axis=1).columns
    data[temp_cols] = F2K(data[temp_cols])

    return data


if __name__ == '__main__':

    data = load_data()
    data = convert_units(data)

    data = data.assign(
        TC_evap_sat=data.apply(
            lambda r: F2K(PropsSI('T', 'P', 6894.76*r['PE1: p_comp_suc'], 'Q', 1.0, 'R407C')), axis=1
        )
    )
    data = data.assign(
        TC_sh=data['TC2: T_comp_suc'] - data['TC_evap_sat'],
    )
    #data=data.assign(
    #        h_2=data.appy(
    #                lambda r:(PropsSI('H','P', 6894.76*r['PE2: p_comp_dis'],'T', K2F))))


    y = 'WT1: P_UUT'

    X = ["TC1",
        "P3", ]
    X = []
    X.extend(list(data.filter(regex="TC[0-5]", axis=1).columns))
    X.extend(list(data.filter(regex="PE", axis=1).columns))
    X = np.array(X)

# Build our regression function first, and then fit it on the data we have.
    RegFunc = ExtraTreesRegressor(n_estimators=100)
    RegFunc.fit(X=data[X].values, y=data[y].values)

    # Now determine which features are most important.
    importances = RegFunc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in RegFunc.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]

#    fig, ax = plt.subplot(1,1)
#    ind=np.arange(21)   #the x locations for the groups
#    width=0.35  #the width of the bars
    
#    # now to plot the figure...
#    plt.figure(figsize=(12, 8))
#    fig = importances.plot(kind='bar')
#    fig.set_title("Amount Frequency")
#    fig.set_xlabel("Amount ($)")
#    fig.set_ylabel("Frequency")
#    fig.set_xticklabels(x_labels)
    
    
    pprint(list(X[indices]))
    plt.bar(range(len(X)), importances)
    plt.ylabel("Importance % [-]")
    plt.xlabel(X)
    plt.show()
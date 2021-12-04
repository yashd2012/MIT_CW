# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Initiatilzing packages

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns 
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error

# Working directory assignment

os.chdir("/Users/yashdixit/Downloads/OceanFlow")

# Average velocity function
def avgvel(UX,UY):
    UXsq = UX.multiply(UX)
    UYsq = UY.multiply(UY)
    Velsq = UXsq + UYsq
    Vel = Velsq.apply(np.sqrt)
    return Vel

# Direction vector of flow field 
def dirn(UX,UY):
    ratio = UY.divide(UX)
    taninv = ratio.apply(np.arctan)
    return taninv

# Reading u, v flow data separately 

USum = pd.read_csv("1u.csv", sep = ',') 
USum[:] = 0
Ugrad = USum

for uindex in range(1,101):
    name = "{index}u.csv".format(index = uindex)
    U = pd.read_csv(name, sep=',')
    U = U.multiply((25/0.9))
    USum = USum + U
    
UAvg = USum / 100

#uax = sns.heatmap(UAvg, cmap = "PiYG")
#uax.invert_yaxis()


VSum = pd.read_csv("1v.csv", sep = ',') 
VSum[:] = 0
Vgrad = VSum

for vindex in range(1,101):
    name = "{index}v.csv".format(index = vindex)
    V = pd.read_csv(name, sep=',')
    V = V.multiply((25/0.9))
    VSum = VSum + V
    
VAvg = VSum / 100
#
#vax = sns.heatmap(VAvg, cmap = "PiYG")
#vax.invert_yaxis()

#plt.show()

# Combined visualization of horizontal and vertical velocities

NetVel = avgvel(UAvg,VAvg)

#vel = sns.heatmap(NetVel, cmap = 'viridis')
#vel.invert_yaxis()
#plt.show()

Dirn = dirn(UAvg,VAvg)
#dr = sns.heatmap(Dirn, cmap = 'Blues')
#dr.invert_yaxis()
#plt.show()

nx, ny = 555, 503
x = np.linspace(0, 555, nx)
y = np.linspace(0, 503, ny)

#plt.streamplot(x,y,UAvg,VAvg)


# Gradient Calculation
#
#for xin in range(1,555):
#    for yin in range(1,503):
#        
#        Ugrad.iloc[yin:yin+1,xin:xin+1] = UAvg.iloc[yin:yin+1,xin+1:xin+2] - UAvg.iloc[yin:yin+1,xin:xin+1]
#        
#        Vgrad.iloc[yin:yin+1,xin:xin+1] = VAvg.iloc[yin+1:yin+2,xin:xin+1] - UAvg.iloc[yin:yin+1,xin:xin+1]
#        
        
        
###   
        
        
        
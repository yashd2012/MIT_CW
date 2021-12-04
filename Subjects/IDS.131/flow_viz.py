#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 21:54:59 2018

@author: yashdixit
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

#loc = np.array([[200,450],[750,200],[1200,1350],[1500,600],[1500,800],[500,600],[250,1200]])
loc = np.random.multivariate_normal([300,1050],[[50, 0], [0, 50]],10)
(m,n) = loc.shape 

plt.figure()

# Initial location in km
for c in range(0,m):
    
    (Xinitial, Yinitial) = loc[c]
        
    stdev = 50 
    # Convert to grid units i.e. Xi'th and Yi'th position in the grid matrix
    
    Xi = Xinitial / 3
    Yi = Yinitial / 3
    
    Xi_grid = int(round(Xi))
    Yi_grid = int(round(Yi))      
    
    # Create empty dataframe to store trajectory coordinates
    
    time_scale = 40
    col = ['XX','YY']
    L = pd.DataFrame(index = range(time_scale), columns = col)
    L[:] = 0      
    
    L.loc[0] = [Xinitial,Yinitial]
    
    Xnew = Xinitial
    Ynew = Yinitial
    
    for tindex in range(1,time_scale):
        
        uname = "{index}u.csv".format(index = tindex)
        vname = "{index}v.csv".format(index = tindex)
        U = pd.read_csv(uname, sep=',')
        V = pd.read_csv(vname, sep=',')
        U = U.multiply((25/0.9))        # U in cm/sec
        V = V.multiply((25/0.9))        # V in cm/sec
        
        # Velocities at grid point
        
        UHolder = []
        VHolder = []
        
        UHolder = U.iloc[ Yi_grid : Yi_grid + 1 , Xi_grid : Xi_grid + 1 ]
        uscalar = UHolder.values
        
        VHolder = V.iloc[ Yi_grid : Yi_grid + 1 , Xi_grid : Xi_grid + 1 ]
        vscalar = VHolder.values
        
        # New trajectory coordinates 
        
        Xnew = ( (Xnew) + ((uscalar*3*60*60)/(100*1000)) )
        Ynew = ( (Ynew) + ((vscalar*3*60*60)/(100*1000)) )
        
        L.loc[tindex] = [Xnew, Ynew]
        
    #
    #L.plot(x='YY',y='XX',style ='o')
    
#    map = pd.read_csv("mask.csv", sep=',')
#    mapmap = sns.heatmap(map)
    #    
    nx = 555*3
    ny = 503*3
    xg = np.linspace(0, 555*3, nx)
    yg = np.linspace(0, 503*3, ny)
    
    #t = L.plot(x='XX',y='YY', xlim = (0, 550), ylim = (0, 503) )
    #t.set_xlim = (0, 550*3)
    #t.set_ylim = (0, 503*3)
   
    map = pd.read_csv("mask.csv", sep=',')
##    map = np.flipud(map)
#    
    plt.imshow(map, aspect = 1, extent = (0,550*3,0,504*3))

    plt.plot(L['XX'],L['YY'], linewidth = 0.3)
    plt.xlim(0,550*3)
    plt.ylim(0,505*3)
    
    
    #mapmap = sns.heatmap(map)
plt.savefig('traj0.jpeg', dpi = 300)
plt.show()

    
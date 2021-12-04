#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 09:41:51 2018

@author: yashdixit
"""

import gzip
import shutil
import os

import numpy as np
import pandas as pd
import csv
from datetime import datetime
from collections import OrderedDict
import matplotlib.pyplot as plty
import collections
import scipy
from scipy import signal, misc

import statsmodels as sm
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import autocorrelation_plot
from pandas.tools.plotting import lag_plot
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA

def smooth(x,window_len=3,window='hanning'):
    
#    if x.ndim != 1:
#        raise ValueError("smooth only accepts 1 dimension arrays")
#        
#    if x.size < window_len:
#        raise ValueError(Input vector needs to be bigger than window size.)


    if window_len<3:
        return x


#    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def smooth1(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

x = np.random.random(40)

plty.plot(x, label = 'x')
plty.plot(smooth(x), label = 'smooth(x)')
plty.plot(smooth1(x,3), label = 'smooth1(x,5)')
plty.plot(np.convolve(x, np.ones((3,))/3, mode='valid'), label = 'conv-valid')
plty.plot(np.convolve(x, np.ones((3,))/3, mode='same'), label = 'conv-same-3')
plty.plot(np.convolve(x, np.ones((5,))/5, mode='same'), label = 'conv-same-5')


plty.legend(loc= 'bottom left')
plty.title('Smoothing Test')
plty.show()


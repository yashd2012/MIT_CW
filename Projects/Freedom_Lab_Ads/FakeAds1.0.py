#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:26:03 2018

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

def smooth1(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def smooth2(x,n):
    
    y = np.convolve(x, np.ones((n,))/n, mode='same')
    
    return y
    
os.chdir("/Users/yashdixit/Downloads/IDS.131 Project")

c1 = "cleveland"
c2 = "boston"
c3 = "oakland"

c12 = "cle-bos"
c23 = "bos-oak"
c31 = "oak-cle"

df = pd.read_csv('geohash_prices_locations.csv')
dg = pd.read_csv('for_final.csv')

fake=[(dg['Fake']==False) & (dg['Fake_num']==False)]
fad = pd.DataFrame(fake)
fad = fad.T
dg['FA']=fad
#print(df.keys())

##### City 2
timeseries_bs = OrderedDict()

for index, row in dg.iterrows():
    # print row
    s = row["date"]
    f = "%Y-%m-%d"
    datetime = datetime.strptime(s,f)
#    if datetime.year == 2018:
#        continue
#    if datetime.month < 5:
#        continue
#    if datetime.month > 7:
#        break
    places = row["top_location"].strip().split(",")
    
    if (c2 in places) & (len(places)<=3) & (row['FA'] == True):
        if datetime not in timeseries_bs:
            timeseries_bs[datetime] = {"price":0, "num":0}
		
#        timeseries_bs[datetime]["price"] += (row["avgRate"])
        timeseries_bs[datetime]["num"] += (row["doc_count"])
    
timeseries_b = OrderedDict()

for index, row in df.iterrows():
    # print row
    s = row["date"]
    f = "%Y-%m-%d"
    datetime = datetime.strptime(s,f)
    if datetime.year == 2017:
        continue
    if datetime.month < 2:
        continue
    if datetime.month > 5:
        break
    places = row["top_location"].strip().split("~")
    
    if c2 in places and len(places)<=3 :
        if datetime not in timeseries_b:
            timeseries_b[datetime] = {"price":0, "num":0}
		
        timeseries_b[datetime]["price"] += (int(row["avgRate"]))
        timeseries_b[datetime]["num"] += (int(row["doc_count"]))

d1 = pd.DataFrame(timeseries_bs, columns=timeseries_bs.keys())
d2 = pd.DataFrame(timeseries_b, columns=timeseries_bs.keys())
d1 = d1.T
d2 = d2.T
y1 = d1['num']
y2 = d2['num']
y2 = y2.dropna()
plty.plot(y2)
plty.show()

hn = np.array(timeseries_bs.items())
at = np.array(timeseries_b.items())
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

def smooth(x,window_len=5,window='hanning'):
    if window_len<3:
        return x

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def correlate_cities(city1, city2, df, startMonth, startYear, endMonth, endYear):
    timeseries1 = series_data(city1, df, startMonth, startYear, endMonth, endYear)
    timeseries2 = series_data(city2, df, startMonth, startYear, endMonth, endYear)

    return (timeseries1, timeseries2)


def plot_correlation(city1, city2, df, startMonth, startYear, endMonth, endYear)
    timeseries1, timeseries2 = correlate_cities(city1, city2, df, startMonth, startYear, endMonth, endYear)

    x1p = timeseries1.keys()
    y1p = [v["num"] for k,v in timeseries_at.items()]
    x2p = timeseries2.keys()
    y2p = [v["num"] for k,v in timeseries_bs.items()]

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(len(x1p)):
        if not np.isnan(y1p[i]) and not np.isnan(y2p[i]):
            x1.append(x1p[i])
            y1.append(y1p[i])
            x2.append(x2p[i])
            y2.append(y2p[i])

    y1s = smooth(y1)
    y2s = smooth(y2)

    k = np.correlate(y1,y2)

    c = scipy.signal.correlate(y1,y2)

    fig, plt = plty.subplots ( nrows=1, ncols=1 )
    plty.xlabel("days")
    plty.ylabel("price of ads seen")
    plty.plot(x1,y1)
    plty.show()
    plty.plot(x2,y2)
    plty.gcf().autofmt_xdate()
    plty.show()

    print(k)
    print(c)
    return k, c


def series_data(city, df, startMonth, startYear, endMonth, endYear):
    timeseries = OrderedDict()
    
    for index, row in df.iterrows():
        s = row["date"]
        f = "%Y-%m-%d"
        d = datetime.strptime(s, f)
        if d.year < startYear or (d.year == startYear and d.month < startMonth) or d.year > endYear or (d.year == startYear and d.month > endMonth):
            continue
        
        places = row["top_location"].strip().split("~")
        
        if city in places and len(places)<=3:
            if datetime not in timeseries:
                timeseries[datetime] = {"price":0, "num":0}
            
            timeseries[datetime]["price"] += (row["avgRate"])
            timeseries[datetime]["num"] += (row["doc_count"])

    return timeseries

def sport_correlations(games, df, startMonth = 1, startYear = 2017, endMonth = 11, endYear = 2018):
    pairings_to_check = set()

    for index, row in games.iterrows():
        if row["Stage"] not in ["Championship", "Super Bowl", "NBA Finals"]:
            continue
        this_pairing = tuple(row["Team 1"], row["Team 2"], row["Host"])
        pairings_to_check.add((this_pairing[0], this_pairing[1]))
        pairings_to_check.add((this_pairing[1], this_pairing[2]))
        pairings_to_check.add((this_pairing[2], this_pairing[0]))

    pairs = set()
    for pair in pairings_to_check():
        if (pair[1], pair[0]) not in pairs:
            pairs.add(pair)

    for pair in pairs:
        plot_correlation(df, pair[0], pair[1], df, startMonth, startYear, endMonth, endYear)



os.chdir("/Users/yashdixit/Downloads/IDS.131 Project")
df = pd.read_csv('geohash_prices_locations.csv')

games = pf.read_csv("games.csv")

sport_correlations(games, df)


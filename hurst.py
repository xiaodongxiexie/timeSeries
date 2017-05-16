# coding: utf-8

from __future__ import division
from collections import Iterable

import numpy as np
from pandas import Series

from numpy import std, subtract, polyfit, sqrt, log

def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""

    # create the range of lag values
    i = len(ts) // 2
    lags = range(2, i)
    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst Exponent from the polyfit output
    return poly[0] * 2.0

def calcHurst2(ts):

    if not isinstance(ts, Iterable):
        print 'error'
        return

    n_min, n_max = 2, len(ts)//3
    RSlist = []
    for cut in range(n_min, n_max):
        children = len(ts) // cut
        children_list = [ts[i*children:(i+1)*children] for i in range(cut)]
        L = []
        for a_children in children_list:
            Ma = np.mean(a_children)
            Xta = Series(map(lambda x: x-Ma, a_children)).cumsum()
            Ra = max(Xta) - min(Xta)
            Sa = np.std(a_children)
            rs = Ra / Sa
            L.append(rs)
        RS = np.mean(L)
        RSlist.append(RS)
    return np.polyfit(np.log(range(2+len(RSlist),2,-1)), np.log(RSlist), 1)[0]


def hurst(history):
    '''只是将时间序列分为1、2、4、8、16、32等份'''
    daily_return = list(100*history.pct_change())[1:]
    # print(daily_return)
    ranges = ['1','2','4','8','16','32']
    lag = Series(index = ranges)
    for i in range(len(ranges)):
        if i==0:
            lag[i] = len(daily_return)
        else:
            lag[i] = lag[0]//(2**i)

    ARS = Series(index = ranges)
    
    for r in ranges:
        #RS用来存储每一种分割方式中各个片段的R/S值
        RS = list()
        #第i个片段 
        for i in range(int(r)):
            #用Range存储每一个片段数据
            Range = daily_return[int(i*lag[r]):int((i+1)*lag[r])]
            sigma = np.std(Range)
            mean = np.mean(Range)
            for k in range(1,len(Range)-1):
                Range[k+1]=Range[k]+Range[k+1]
            
            Deviation = Range - mean
            maxi = max(Deviation)
            mini = min(Deviation)
            RS.append(maxi - mini)
            # sigma = np.std(Range)
            RS[i] = RS[i]/sigma
            # print(maxi-mini)
            # print(Range)
        ARS[r] = np.mean(RS)
        # print(RS)
    lag = np.log10(lag)
    ARS = np.log10(ARS)
    hurst_exponent = np.polyfit(lag,ARS,1)
    # hurst_exponent = (sm.OLS(ARS.values,lag.values)).fit()
    hurst = hurst_exponent[0]
    return hurst

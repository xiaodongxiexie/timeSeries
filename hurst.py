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
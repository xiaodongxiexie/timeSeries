#coding: utf-8

from statsmodels.tsa.stattools import adfuller
from pandas import Series, DataFrame


def testStationarity(timeSer):
    '''
    检测时间序列稳定性，如稳定则继续，否则继续差分。
    '''
    dftest = adfuller(timeSer)
    dfoutput = Series(dftest[:4],index=['Test Statistics','p-value','lags','nobs'])
    for key,value in dftest[4].items():
        dfoutput['Critical values(%s)'%key] = value
    if dfoutput['Test Statistics'] < dfoutput['Critical values(5%)'] and dfoutput['p-value']<0.1:
        stationarity = True
    else:
        stationarity = False
    return stationarity

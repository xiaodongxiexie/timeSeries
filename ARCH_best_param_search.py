# -*- coding: utf-8 -*-
# @Author: xiaodong


import sys
import os
import time

import arch
import numpy as np 
import pandas as pd 
import random 
#import statsmodels.api as sm 

from functools import wraps
from pandas import Series, DataFrame
#from statsmodels.graphics.api import qqplot 
#from statsmodels.graphics.tsaplots import plot_pacf 
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller 
from statsmodels.tsa.stattools import acf, pacf  
#from statsmodels.tsa.seasonal import seasonal_decompose
#from scipy import stats 


def decorator(func):
	'''
	define a wrapped-function for detecting the given function costs
	'''
	@wraps(func)
	def wrapped(*args,**kwargs):

		_start = time.clock()
		_result = func(*args,**kwargs)
		_end = time.clock()
		print func.__name__, u'共耗时： ', _end - _start,'\n'  
		return _result 

	return wrapped


@decorator
def concatFile(path):

	data = DataFrame()
	for _file in os.listdir(path):
		if '.csv' in _file:
			data = pd.concat([data, pd.read_csv(os.path.join(path, _file))])

	ser = data[['Open', 'datetime_stp']].set_index('datetime_stp')
	ser.index = pd.to_datetime(ser.index)

	global dateList
	dateList = ser.Open.groupby(lambda x:x.date()).size().index
	return ser.Open


@decorator
def productDiff(timeSer,diff,date):

	ser = timeSer
	ser = ser[date].diff(diff).dropna()
	if len(ser) > 100:
		sign = Series(map(str,map(lambda x:x[:4],map(str,(ser.index.map(lambda x:x.time()))))))
		ser = ser[(('15:01'>sign) &(sign>'09:20)')).ravel()]
		i = random.choice(range(100,len(ser)-10,10))

		return ser[:i] 
	else:
		return 


@decorator
def testStationarity(timeSer):
    dftest = adfuller(timeSer)
    dfoutput = Series(dftest[:4],index=['Test Statistics','p-value','lags','nobs'])

    for key,value in dftest[4].items():
        dfoutput['Critical values(%s)'%key] = value

    if dfoutput['Test Statistics'] < dfoutput['Critical values(5%)']:

    	if dfoutput['p-value']<0.1:
	        stationarity = True	      
    else:
        stationarity = False       

    return stationarity


@decorator
def p_q_choice(timeSer,nlags=20,alpha=.05):

        acf_x,confint = acf(timeSer,nlags=nlags,alpha=alpha)
        acf_px, confint2 = pacf(timeSer,nlags=nlags,alpha=alpha)

        confint = confint - confint.mean(1)[:,None]    
        confint2 = confint2 - confint2.mean(1)[:,None] 

        acf_x,confint = acf_x[1:], confint[1:]
        acf_px, confint2 = acf_px[1:], confint2[1:]

        for key2, value2 in enumerate(acf_px):
        	if (np.abs(acf_px[key2:]) < np.abs(confint2[:,0])[key2:]).all():
        		p = key2 
        		if p == 0 or p > 10:
        			return errorHandle(timeSer)
        		break
        else:
        	return errorHandle(timeSer)

        return p

def errorHandle(timeSer):

	best_aic = sys.maxint
	for i in [3,5,7,8]:
		current_aic = ARMA(timeSer,order=(i,0)).fit(disp=0).aic
		p = i if current_aic < best_aic else p
	return p


@decorator
def modelARCH(timeSer,i=10,diff=1,date='2016-12-19'):

	ser = timeSer
	train = ser[:-i]
	test = ser[-i:]

	p  = p_q_choice(ser)
	resid2 = np.square(ARMA(ser, (p,0)).fit(disp=0).resid)

	p2 = p_q_choice(resid2)

	arch_model = arch.arch_model(train, mean='AR',lags=p, vol='ARCH', p=p2)

	resid = arch_model.fit(update_freq=0, disp=0)
	count = len(train)
	pred = resid.forecast(horizon=10, start=count-1).mean.iloc[count-1]

	df1 = DataFrame({'pred':pred.cumsum()})
	df2 = DataFrame({'real':test.cumsum()})
	df2.index = df1.index = range(len(df1))

	return df1.join(df2)

if __name__ == '__main__':
    
	error = 0
	path = r'D:\work store private\test'
	data = DataFrame()

	for loop_num in range(10):
		timeSer = concatFile(path)
		for date in map(str, dateList):
			for diff in range(11,1,-1):
				train_ser = productDiff(timeSer, diff, date)
				if train_ser is not None:
					if testStationarity(train_ser):
						print date
						try:
							df = modelARCH(train_ser,i=10,diff=diff,date=date)
							data = pd.concat([data, df])

							data.to_csv(r'C:\Users\Administrator\Desktop\ARCH_para.csv')  
                                                                
						except:
							error += 1
							print error  
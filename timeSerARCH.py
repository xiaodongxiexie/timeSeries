# -*- coding: utf-8 -*-
# @Author: xiaodong


import sys
import os
import time

import arch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
#import statsmodels.api as sm

from functools import wraps
from pandas import Series, DataFrame
from threading import Thread
#from statsmodels.graphics.api import qqplot
from statsmodels.graphics.tsaplots import plot_pacf
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
    def wrapped(*args, **kwargs):

        _start = time.clock()
        # print u'开始时间：', _start
        print u'此处开始调用函数：' + func.__name__
        _result = func(*args, **kwargs)
        _end = time.clock()
        # print u'结束时间：', _end
        print func.__name__, u'共耗时： ', _end - _start, '\n'

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
    dateList = ser.Open.groupby(lambda x: x.date()).size().index

    return ser.Open


@decorator
def productDiff(path, diff=1, date='2016-12-19'):

    ser = concatFile(path)
    ser = ser[date].diff(diff).dropna()
    if len(ser) > 100:
        sign = Series(map(str, map(lambda x: x[:4], map(
            str, (ser.index.map(lambda x: x.time()))))))
        ser = ser[(('15:01' > sign) & (sign > '09:20)')).ravel()]
        i = random.choice(range(100, len(ser) - 10, 10))

        return ser[:i]
    else:
        print u'可用序列长度过短，即将报错。'


@decorator
def testStationarity(timeSer):
    '''
    test the stationarity of the time series by continue diff it 
    '''
    dftest = adfuller(timeSer)
    dfoutput = Series(dftest[:4], index=[
                      'Test Statistics', 'p-value', 'lags', 'nobs'])

    for key, value in dftest[4].items():
        dfoutput['Critical values(%s)' % key] = value

    if dfoutput['Test Statistics'] < dfoutput['Critical values(5%)']:

        print dfoutput['p-value']

        if dfoutput['p-value'] < 0.1:
            stationarity = True
            print u'it is ok!'
    else:
        stationarity = False
        print u'attention: not ok!'

    return stationarity, dfoutput


@decorator
def p_q_choice(timeSer, nlags=20, alpha=.05):

    acf_x, confint = acf(timeSer, nlags=nlags, alpha=alpha)
    acf_px, confint2 = pacf(timeSer, nlags=nlags, alpha=alpha)

    confint = confint - confint.mean(1)[:, None]
    confint2 = confint2 - confint2.mean(1)[:, None]

    acf_x, confint = acf_x[1:], confint[1:]
    acf_px, confint2 = acf_px[1:], confint2[1:]

    # for key1,value1 in enumerate(acf_x):
    # 	if (np.abs(acf_x[key1:]) < np.abs(confint[:,0])[key1:]).all():
    # 		q = key1
    # 		break
    # else:
    # 	return errorHandle(timeSer)

    for key2, value2 in enumerate(acf_px):
        if (np.abs(acf_px[key2:]) < np.abs(confint2[:, 0])[key2:]).all():
            p = key2
            if p == 0 or p > 10:
                return errorHandle(timeSer)
            break
    else:
        return errorHandle(timeSer)

    return p


def errorHandle(timeSer):
    print 'in here'

    best_aic = sys.maxint
    for i in [3, 5, 7, 8]:
        current_aic = ARMA(timeSer, order=(i, 0)).fit(disp=0).aic
        p = i if current_aic < best_aic else p
        print 'in here2'

    return p


def close(count):
	time.sleep(count)
	plt.close()


@decorator
def modelAR(timeSer, path, diff, date, lags=20, showFig=True):

    p = p_q_choice(timeSer)
    print 'check here1'

    ser = productDiff(path, diff, date)
    print 'check here2'

    #L = []
    # if p >9:
    # 	order = (p,0)
    # else:
    # 	best_aic = sys.maxint
    # 	for i in range(1,9):
    # 		current_aic = ARMA(ser,order=(i,0)).fit(disp=0).aic
    # 		L.append(current_aic)
    # 		p = i if current_aic < best_aic else p
    # 		order = (p,0)
    # print L
    order = (p, 0)
    print 'check here3'
    print p

    model = ARMA(ser, order).fit(disp=0)
    print 'check here4'

    resid = model.resid

    global resid2
    resid2 = np.square(resid)

    thread = Thread(target=close, args=(5,))
    thread2 = Thread(target=close, args=(10,))
    thread.start()
    thread2.start()

    if showFig:
        fig = plt.figure(figsize=(20, 6))
        ax1 = fig.add_subplot(211)
        fig = plot_pacf(ser, lags=lags, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = plot_pacf(resid2, lags=lags, ax=ax2)

        plt.figure(figsize=(20, 12))
        plt.subplot(211)
        plt.plot(resid, label='resid')
        plt.legend()

        plt.subplot(212)
        plt.plot(resid2, label='resid ** 2')
        plt.legend(loc=0)
        plt.show()

    _acf, q, p = acf(resid2, nlags=25, qstat=True)
    out = np.c_[range(1, 26), _acf[1:], q, p]
    output = DataFrame(out, columns=['lag', 'AC', 'Q', 'P-value'])
    output = output.set_index('lag')

    return output


@decorator
def modelARCH(timeSer, i=10, summary=False,
              params=False, diff=1, date='2016-12-19'):

    ser = productDiff(path, diff, date)
    train = ser[:-i]
    test = ser[-i:]

    p = p_q_choice(timeSer)
    p2 = p_q_choice(resid2)

    arch_model = arch.arch_model(train, mean='AR', lags=p, vol='ARCH', p=p2)

    resid = arch_model.fit(update_freq=0, disp=0)
    if summary:
        print resid.summary()
    if params:
        print resid.params
    # plt.figure(1,figsize=(10,6))

    
    thread = Thread(target=close, args=(5,))
    thread2 = Thread(target=close, args=(10,))
    thread.start()
    thread2.start()
    fit_plot = resid.hedgehog_plot()

    count = len(train)
    pred = resid.forecast(horizon=10, start=count - 1).mean.iloc[count - 1]

    df1 = DataFrame({'pred': pred.cumsum()})
    df2 = DataFrame({'real': test.cumsum()})
    df2.index = df1.index = range(len(df1))

    print df1.join(df2)

    kwargs = dict(alpha=.5, width=.3)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.bar(range(i), test.values, label='real value', color='green', **kwargs)
    plt.axhline(0, ls='--', lw=1.5, color='green')

    pred.plot(label='predict value', color='red',
              ax=ax.twinx(), kind='bar', **kwargs)
    plt.axhline(y=0, ls='--', color='red')

    title = '\n'.join([date, 'p=%s' % p, 'p2=%s' % p2])

    plt.title(title)

    plt.show()

if __name__ == '__main__':

    path = r'D:\work store private\test'

    diff = 1
    lags = 40
    concatFile(path)
    flag = True
    while flag:
        ask = raw_input(unicode('是否随机选择一天：(y/n)', 'utf-8').encode('gbk'))
        if 'y' in ask:
            date = random.choice(map(str, dateList))
            print u'选择日期为： ', date
        else:
            print dateList
            date = raw_input(unicode('请输入一个有效日期：', 'utf-8').encode('gbk'))
            if date.strip() in map(str, dateList):
                date = date.strip()
            else:
                print '\n', '!' * 50
                print u'输入日期不对，请走点心, OK?'.center(30, '*'), '\n'
                continue

        for diff in range(10, 1, -1):
            timeSer = productDiff(path, diff=diff, date=date)
            if testStationarity(timeSer)[0]:
                print 'diff= ', diff, '\n'
                break
        else:
            print 'error here'

        args = [timeSer, path]
        kwargs = {'diff': diff,
                  'date': date,
                  'lags': lags}
        try:
            modelAR(*args, **kwargs)
            modelARCH(timeSer, date=date)

            flag = raw_input(unicode('是否退出：(y/n)', 'utf-8').encode('gbk'))

            if 'y' in flag:
                flag = False
            else:
                flag = True
        except:
            continue

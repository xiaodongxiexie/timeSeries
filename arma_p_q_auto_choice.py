#encoding:utf-8

import sys 
import numpy as np

from statsmodels.tsa.stattools import acf, pacf, adfuller


#用来根据给定的滞后阶数自动选择p，q值，并进行传递
def proper_model(data_ts,maxLag):
	init_bic = sys.maxint
	init_p = 0
	init_q = 0
	init_properModel = None
	for p in np.arange(maxLag):
		for q in np.arange(maxLag):
			model = ARMA(data_ts,order=(p,q))
			try:
				results_ARMA = model.fit(disp=-1,method='css')
			except:
				continue
			bic = results_ARMA.bic 
			if bic < init_bic:
				init_p = p
				init_q = q 
				init_properModel = results_ARMA
				init_bic = bic 
	return init_bic,init_p,init_q,init_properModel


def p_q_choice(timeSer,nlags=40,alpha=.05):
    '''
    根据可信区间用来查找最佳p,q值，
    默认为选择95%可信区间，可修改alpha参数来改变
    '''
    acf_x,confint = acf(timeSer,nlags=nlags,alpha=alpha)
    acf_px, confint2 = pacf(timeSer,nlags=nlags,alpha=alpha)

    confint = confint - confint.mean(1)[:,None]    #用来计算acf可信度上下区间
    confint2 = confint2 - confint2.mean(1)[:,None] #用来计算pacf可信度上下区间

    #当出现第一个满足可信区间的值时则停止查找
    #这是因为根据历史数据测试时发现如第1个值满足，则其后值收敛，取第1个值即可
    for key1,x,y,z in zip(range(nlags),acf_x,confint[:,0],confint[:,1]):
        if x > y and x < z:
            q = key1
            break
    for key2,x,y,z in zip(range(nlags),acf_px,confint2[:,0],confint2[:,1]):
        if x > y and x < z:
            p = key2
            break
    return p, q 

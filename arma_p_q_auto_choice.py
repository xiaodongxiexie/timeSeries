#encoding:utf-8

import sys 
import numpy as np


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
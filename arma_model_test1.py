#coding:utf-8


import pandas as pd 
import numpy as np 
import sys
import matplotlib.pyplot as plt 
from statsmodels.tsa.arima_model import ARMA 
from dateutil.relativedelta import relativedelta 
from copy import deepcopy 


class arima_model:

	def __init__(self, ts, maxLag=9):
		self.data_ts = ts 
		self.resid_ts = None
		self.predict_ts = None
		self.maxLag = maxLag 
		self.p = maxLag 
		self.q = maxLag 
		self.properModel = None 
		self.bic = sys.maxint 

	def get_proper_model(self):
		self._proper_model()
		self.predict_ts = deepcopy(self.propermModel.predict())
		self.resid_ts = deepcopy(self.properModel.resid)

	def _proper_model(self):
		for p in np.arange(self.maxLag):
			for q in np.arange(self.maxLag):
				model = ARMA(self.data_ts,order=(p,q))
				try:
					results_ARMA = model.fit(disp=-1,method='css')
				except:
					continue
				bic = results_ARMA.bic
				if bic < self.bic:
					self.p = p 
					self.q = q
					self.properModel = results_ARMA 
					self.bic = bic 
					self.resid_ts = deepcopy(self.properModel.resid)
					self.predict_ts = self.properModel.predict()

	def certain_model(self,p,q):
		model = ARMA(self.data_ts,order=(p,q))
		try:
			self.properModel = model.fit(disp=-1,method='css')
			self.p = p
			self.q = q
			self.bic = self.properModel.bic 
			self.predict_ts = self.properModel.predict()
			self.resid_ts = deepcopy(self.properModel.resid)
		except:
			print u'p,q有问题'

	def forecast_next_day_value(self,type='day'):
		self.properModel.forecast()
		if self.data_ts.index[-1] != self.resid_ts.index[-1]:
			raise ValueError('error')
		if not self.properModel:
			raise ValueError('error2')
		para = self.properModel.params

		if self.p == 0:
			ma_value = self.resid[-self.q:]
			values = ma_value.reindex(index=ma_value.index[::-1])
		elif self.q == 0:
			ar_value = self.data_ts[-self.p:]
			values = ar_value.reindex(index=ar_value.index[::01])
		else:
			ar_value = self.data_ts[-self.p:]
			ar_value = ar_value.reindex(index=ar_value.index[::-1])
			ma_value = self.resid_ts[-self.q:]
			ma_value = ma_value.reindex(index=ma_value.index[::-1])
			values = ar_value.append(ma_value)

		predict_value = np.dot(para[1:],values) + self.properModel.constant[0]
		self._add_new_data(self.predict_ts,predict_value,type)
		return predict_value 

	def _add_new_data(self, ts, dat, type='day'):
		if type == 'day':
			new_index = ts.index[-1] + relativedelta(days=1)
		elif type == 'month':
			new_index = ts.index[-1] + relativedelta(months=1)

		ts[new_index] = dat 

	def add_today_data(self, dat, type='day'):
		self._add_new_data(self.data_ts, dat, type)
		if self.data_ts.index[-1] != self.predict_ts.index[-1]:
			raise ValueError('err')
		self._add_new_data(self.resid_ts,self.data_ts[-1],type)







#coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from dateutil.parser import parse

#%matplotlib inline

data = pd.read_csv(r'C:\Users\Administrator\Desktop\test.csv')



dateIndex = data.groupby(data.datetime_stp.map(lambda x:parse(x).date())).size().index

for i,dateParse in enumerate(dateIndex):
    dataAnalyse = data[data.datetime_stp.map(lambda x:parse(x).date() == parse(str(dateParse)).date())]
    dataSort = dataAnalyse.sort_values(by='datetime_stp')
    dataClean = dataSort[(dataSort.datetime_stp.map(lambda x:int(str(x)[11:13])>8))&(dataSort.datetime_stp.map(lambda x:int(str(x)[11:13])<15))]
    dataClose = dataClean.Close
    
    fig = plt.figure(i,figsize=(12,4))
    
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.set_title(dateParse)
    ax2.set_title(dateParse)
    num, section = ax1.hist(dataClose.diff().dropna(),color='red', alpha=0.6)[:2]
    ax2.pie(num,labels=[(round(section[i],2),round(section[i+1],2)) for i in range(len(section)) if i < len(section)-1],autopct='%%%.2f')
    plt.show()

def statisticDiff(path,diff=1,col='Close'):
    #读取指定路径文件
    data = pd.read_csv(path)
    dataIndex = data.groupby(data.datetime_stp.map(lambda x:parse(x).date())).size().index
    diffArr = np.array([])
    for i,dateParse in enumerate(dateIndex):
        dataAnalyse = data[data.datetime_stp.map(lambda x:parse(x).date() == parse(str(dateParse)).date())]
        dataSort = dataAnalyse.sort_values(by='datetime_stp')
        dataClean = dataSort[(dataSort.datetime_stp.map(lambda x:int(str(x)[11:13])>8))&(dataSort.datetime_stp.map(lambda x:int(str(x)[11:13])<15))]
        dataClose = dataClean[col]

        diffArr = np.hstack((dataClose.diff(diff).dropna(),diffArr))
    return diffArr

def plotDiff(arr,diff=1,fig=1):
    fig = plt.figure(fig,figsize=(12,4))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.set_title('diff=%s'%diff)
    ax2.set_title('diff=%s'%diff)
    
    num, section = ax1.hist(arr,color='red', alpha=0.6)[:2]

    ax2.pie(num,labels=[(round(section[i],2),round(section[i+1],2)) for i in range(len(section)) if i < len(section)-1],autopct='%%%.2f')
    plt.show()

path = 'C:\\Users\\Administrator\\Desktop\\test.csv'

L = range(1,16)
L.extend(range(20,65,5))
L.extend(range(90,210,30))

for i,diff in enumerate(L):
    plotDiff(statisticDiff(path,diff=diff),diff=diff,fig=i)

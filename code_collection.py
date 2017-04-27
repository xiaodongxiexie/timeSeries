#coding:utf-8

##############################用来将txt转换成series###########################
def text_to_ser(path):
    import codecs
    from pandas import Series
    List = []
    Lindex = []
    Lvalues = []
    with codecs.open(path,'r','gbk') as f:
        data = f.readlines()
    for line in data:
        if 'predict_value' in line:
            List.append(line)    
    for L in List:
        Lindex.append(L[:8])
        Lvalues.append(L.split('predict_value')[1].strip())
    ser = Series(Lvalues,index=Lindex)
    return ser



##############################作图用1###########################
import pandas as pd 
import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from pandas import Series, DataFrame
from datetime import datetime
import random

plt.rcParams['font.sans-serif'] = ['SimHei']  #用来显示中文
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

datestr = datetime.now().strftime('%Y%m%d%H%M%S')
fig = plt.figure(1,figsize=(15,5))
ax = fig.add_subplot(1,1,1)

data = pd.read_csv('2.csv')
dataNeed = DataFrame({u'最高价':data['High'],u'最低价':data['Low']})
dataNeed.plot(ax=ax,linewidth=2.0)
_ = ax.scatter(range(len(data.High))[::2], (['3338.6']*len(data.High))[::2], 2, color='r')   #这个值根据同花顺网站调整

def getData():
    dataBuy  = []
    dataSell = []
    data1 = pd.read_csv('1.csv',encoding="gb2312")   #成交记录数据
    data2 = pd.read_csv('2.csv')   #今日股指数据
    for time1,index in zip(data2.date,data2.index):
        for time2,price,panduan in zip(data1.tradeTime,data1.price,data1.offset):
            if time1[:-3] == time2[:-3]:
                if panduan == u'平仓':
                    dataSell.append((index,price))
                else:
                    dataBuy.append((index,price))
    return dataSell, dataBuy

#dataType = ['1','2','1','2','2','1']     #以1代表卖空仓平空仓，2代表建多仓平多仓
data1 = pd.read_csv('1.csv',encoding="gb2312") #此处必须加上encoding才能显示中文
data11 = data1.sort_index(by='tradeTime')
# print data11
dataType = []
for todayDirection,todayOffset in zip(data11.direction,data11.offset):
    if todayDirection == u'多' and todayOffset == u'开仓':
        dataType.append('2')
    elif todayDirection == u'空' and todayOffset == u'开仓':
        dataType.append('1')
# print dataType

dataSell = getData()[0]
dataBuy = getData()[1]
# print dataSell
# print dataBuy

for datasell,datatype in zip(dataSell,dataType):
    if datatype == '1':
        color,annotateText = 'red', 'cover'
    else:
        color,annotateText = 'black', 'sell'
    ax.annotate(annotateText, color=color, xy=datasell, xycoords='data',
                xytext=(-30,+40), textcoords='offset points',
                fontsize=20, arrowprops=dict(arrowstyle='->', color='red', linewidth=2.5))


for databuy,datatype in zip(dataBuy,dataType):
    if datatype == '1':
        color,annotateText = 'red', 'short'
    else:
        color,annotateText = 'black', 'buy'

    ax.annotate(annotateText, color=color, xy=databuy, xycoords='data',
                xytext=(+0,-50), textcoords='offset points',
                fontsize=20, arrowprops=dict(arrowstyle='->', linewidth=2.5))

ax.set_title('VNPY Trade Statistics')
ax.set_xlabel('datetime: %s'% datestr[:8])
ax.set_ylabel('price')  
ax.set_xticks((data.index.values[1::20]))
#ax.set_xticks((data.index.values[1::20])[:-1])   #如果最后一个的时间点不对则用这两个
#ax.set_xticklabels(([date[:5] for date in ((data['date'][1::20]).values)])[:-1], rotation=30)
ax.set_xticklabels(([date[:5] for date in ((data['date'][1::20]).values)]), rotation=30)
print (data['date'][::20]).values

# plt.ylim(3310,3380)
plt.legend(loc='upper left')
plt.text(2, 3339, '3338.6')
plt.grid()  #显示网格
# plt.autoscale(tight=True)
plt.savefig('%s.png'%datestr, bbox_inches='tight')
# plt.show()


######以下为用100阶多项式拟合的曲线################
dataPolyFitNeed = Series(data.High)#[70:80]
x = dataPolyFitNeed.index
y = dataPolyFitNeed.values
fp1, residuals, rank, sv, rcond = sp.polyfit(x,y,120,full=True)
f1 = sp.poly1d(fp1)
fx = sp.linspace(0,x[-1],1000)
# def error(f,x,y):
#   return sp.sum((f(x)-y)**2)
pngIndex = datestr + str(random.randint(0,1000))
fig2 = plt.figure(2,figsize=(15,5))
ax2 = fig2.add_subplot(1,1,1)
ax2.set_title(u'用多项式对交易进行拟合')
ax2.set_xlabel('datetime: %s'% datestr[:8])
ax2.set_ylabel('price') 
ax2.set_xticks((data.index.values[1::20]))
ax2.scatter(range(len(data.High))[::2], (['3338.6']*len(data.High))[::2], 2, color='green') 
ax2.set_xticklabels(([date[:5] for date in ((data['date'][1::20]).values)]), rotation=30)

plt.text(2, 3339, '3338.6')
plt.grid()  #显示网格
plt.plot(fx,f1(fx),linewidth=2.5,color='red',linestyle='-')
plt.autoscale(tight=True)
plt.savefig('%s.png'%pngIndex, bbox_inches='tight')
plt.show((fig,fig2))



##############################作图用2###########################
#-*- encoding: utf-8 -*-


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from pandas import Series, DataFrame
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']  #用来显示中文
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

datestr = datetime.now().strftime('%Y%m%d%H%M%S')
fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(1,1,1)

data = pd.read_csv('2.csv')
dataNeed = DataFrame({u'最高价':data['High'],u'最低价':data['Low']})
dataNeed.plot(ax=ax,linewidth=2.0)
_ = ax.scatter(range(len(data.High))[::2], (['3351.8']*len(data.High))[::2], 2, color='r')   #这个值根据同花顺网站调整

def getData():
    dataBuy  = []
    dataSell = []
    data1 = pd.read_csv('1.csv')   #成交记录数据
    data2 = pd.read_csv('2.csv')   #今日股指数据
    for time1,index in zip(data2.date,data2.index):
        for time2,price,panduan in zip(data1.tradeTime,data1.price,range(len(data1.price))):
            if time1[:-2] == time2[:-2]:
                if (panduan+1) % 2 == 0:
                    dataSell.append((index,price))
                else:
                    dataBuy.append((index,price))
    return dataSell, dataBuy

#dataType = ['1','2','1','2','2','1']    #以1代表卖空仓平空仓，2代表建多仓平多仓
data1 = pd.read_csv('1.csv',encoding="gb2312") #此处必须加上encoding才能显示中文
data11 = data1.sort_index(by='tradeTime')
dataType = []
for todayDirection,todayOffset in zip(data11.direction,data11.offset):
    if todayDirection == u'多' and todayOffset == u'开仓':
        dataType.append('2')
    elif todayDirection == u'空' and todayOffset == u'开仓':
        dataType.append('1')
print dataType

dataSell = getData()[0]
dataBuy = getData()[1]
for datasell,datatype in zip(dataSell,dataType):
    if datatype == '1':
        color,annotateText = 'red', 'cover'
    else:
        color,annotateText = 'black', 'sell'
    ax.annotate(annotateText, color=color, xy=datasell, xycoords='data',
                xytext=(-30,+30), textcoords='offset points',
                fontsize=20, arrowprops=dict(arrowstyle='->', color='red', linewidth=2.5))


for databuy,datatype in zip(dataBuy,dataType):
    if datatype == '1':
        color,annotateText = 'red', 'short'
    else:
        color,annotateText = 'black', 'buy'

    ax.annotate(annotateText, color=color, xy=databuy, xycoords='data',
                xytext=(+0,-40), textcoords='offset points',
                fontsize=20, arrowprops=dict(arrowstyle='->', linewidth=2.5))

ax.set_title('VNPY Trade Statistics')
ax.set_xlabel('datetime: %s'% datestr[:8])
ax.set_ylabel('price')  
ax.set_xticks((data.index.values[1::20]))
#ax.set_xticks((data.index.values[1::20])[:-1])   #如果最后一个的时间点不对则用这两个
#ax.set_xticklabels(([date[:5] for date in ((data['date'][1::20]).values)])[:-1], rotation=30)
ax.set_xticklabels(([date[:5] for date in ((data['date'][1::20]).values)]), rotation=30)
print (data['date'][::20]).values

# plt.ylim(3310,3380)
plt.legend(loc='upper left')
plt.text(2, 3352, '3351.8')
plt.grid()  #显示网格
# plt.autoscale(tight=True)
plt.savefig('%s.png'%datestr, bbox_inches='tight')
plt.show()


##############################作图用3###########################
#-*- encoding: utf-8 -*-


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from pandas import Series, DataFrame
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']  #用来显示中文
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

datestr = datetime.now().strftime('%Y%m%d%H%M%S')
fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(1,1,1)

data = pd.read_csv('2.csv')
dataNeed = DataFrame({u'最高价':data['High'],u'最低价':data['Low']})
dataNeed.plot(ax=ax,linewidth=2.0)
_ = ax.scatter(range(len(data.High))[::2], (['3351.8']*len(data.High))[::2], 2, color='r')   #这个值根据同花顺网站调整

def getData():
    dataBuy  = []
    dataSell = []
    data1 = pd.read_csv('1.csv')   #成交记录数据
    data2 = pd.read_csv('2.csv')   #今日股指数据
    for time1,index in zip(data2.date,data2.index):
        for time2,price,panduan in zip(data1.tradeTime,data1.price,range(len(data1.price))):
            if time1[:-2] == time2[:-2]:
                if (panduan+1) % 2 == 0:
                    dataSell.append((index,price))
                else:
                    dataBuy.append((index,price))
    return dataSell, dataBuy

#dataType = ['1','2','1','2','2','1']    #以1代表卖空仓平空仓，2代表建多仓平多仓
data1 = pd.read_csv('1.csv',encoding="gb2312") #此处必须加上encoding才能显示中文
data11 = data1.sort_index(by='tradeTime')
dataType = []
for todayDirection,todayOffset in zip(data11.direction,data11.offset):
    if todayDirection == u'多' and todayOffset == u'开仓':
        dataType.append('2')
    elif todayDirection == u'空' and todayOffset == u'开仓':
        dataType.append('1')
print dataType

dataSell = getData()[1]
dataBuy = getData()[0]
for datasell,datatype in zip(dataSell,dataType):
    if datatype == '1':
        color,annotateText = 'red', 'cover'
    else:
        color,annotateText = 'black', 'sell'
    ax.annotate(annotateText, color=color, xy=datasell, xycoords='data',
                xytext=(-30,+30), textcoords='offset points',
                fontsize=20, arrowprops=dict(arrowstyle='->', color='red', linewidth=2.5))


for databuy,datatype in zip(dataBuy,dataType):
    if datatype == '1':
        color,annotateText = 'red', 'short'
    else:
        color,annotateText = 'black', 'buy'

    ax.annotate(annotateText, color=color, xy=databuy, xycoords='data',
                xytext=(+0,-40), textcoords='offset points',
                fontsize=20, arrowprops=dict(arrowstyle='->', linewidth=2.5))

ax.set_title('VNPY Trade Statistics')
ax.set_xlabel('datetime: %s'% datestr[:8])
ax.set_ylabel('price')  
ax.set_xticks((data.index.values[1::20]))
#ax.set_xticks((data.index.values[1::20])[:-1])   #如果最后一个的时间点不对则用这两个
#ax.set_xticklabels(([date[:5] for date in ((data['date'][1::20]).values)])[:-1], rotation=30)
ax.set_xticklabels(([date[:5] for date in ((data['date'][1::20]).values)]), rotation=30)
print (data['date'][::20]).values

# plt.ylim(3310,3380)
plt.legend(loc='upper left')
plt.text(2, 3352, '3351.8')
plt.grid()  #显示网格
# plt.autoscale(tight=True)
plt.savefig('%s.png'%datestr, bbox_inches='tight')
plt.show()


##############################统计胜率用###########################
# coding:utf-8


from __future__ import division
import numpy as np
import pandas as pd
from pandas import Series
from matplotlib import pyplot as plt
#%matplotlib inline
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来显示中文
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

#data = pd.read_csv('1.csv',encoding='gb2312')  #读取交易记录，进行统计
data = pd.read_csv('1.csv',encoding='gb2312',parse_dates=True)
myPrice = []
myDirection = []
for x,y in zip(data.price,data.direction):
    myPrice.append(x)
    myDirection.append(y)
myPrice = myPrice[::-1]
myDirection = myDirection[::-1]
winPointsSum = []
lostPointsSum =[]
countDuo = 0
countKong = 0
countKongWin = 0
countKongLost = 0
countPing = 0
countDuoWin = 0
countDuoLost = 0
tongji = []
print u'今日交易次数：',int(len(myPrice)/2)
for price,index,direction in zip(myPrice,range(len(myPrice)),myDirection):
    if index % 2 == 0:
        if direction == u'空':
            countKong += 1
            if price - myPrice[index+1] > 0:
                print u'第%s笔'%(int(index/2 + 1)),u'买空赚了',price - myPrice[index+1],u'个点，',u'胜；'
                tongji.append(price-myPrice[index+1])
                winPointsSum.append(price-myPrice[index+1])
                countKongWin += 1
            elif price - myPrice[index+1] == 0:
                countPing += 1
                tongji.append(0)
            else:
                print u'第%s笔'%(int(index/2 + 1)),u'买空赔了',-(price - myPrice[index+1]),u'个点，', u'败；'
                tongji.append(price-myPrice[index+1])
                lostPointsSum.append(price - myPrice[index+1])
                countKongLost += 1
        if direction == u'多':
            countDuo += 1
            if price - myPrice[index+1] > 0:
                print u'第%s笔'%(int(index/2 + 1)),u'买多赔了',price - myPrice[index+1],u'个点，', u'败；'
                tongji.append(-(price-myPrice[index+1]))
                lostPointsSum.append(-(price - myPrice[index+1]))
                countDuoLost += 1
            elif price - myPrice[index+1] ==0:
                countPing += 1
                tongji.append(0)
            else:
                print u'第%s笔'%(int(index/2 + 1)),u'买多赚了',-(price - myPrice[index+1]),u'个点，', u'胜；'
                tongji.append(-(price-myPrice[index+1]))
                winPointsSum.append(-(price-myPrice[index+1]))
                countDuoWin +=1
    else:
        pass
#for win in winPointsSum:
#    print '%.2f'% win
winplot = ['%.2f'% win for win in winPointsSum]
lostplot = ['%.2f'% lost for lost in lostPointsSum]
winPointsSum = np.array(winPointsSum).sum()
lostPointsSum = np.array(lostPointsSum).sum()
print u'今日胜点数：',winPointsSum
print u'今日败点数：',lostPointsSum

print u'今日建多仓次数：',countDuo
print u'今日监空仓次数：',countKong
print u'建多仓胜次数：',countDuoWin
print u'建空仓胜次数：',countKongWin
print u'平次数：',countPing
print u'今日胜率：',(countKongWin + countDuoWin) / (len(myPrice)/2)
#print tongji
lost = Series(tongji)[[0,1,4,5]]
new_lost = Series(lost,index=range(6))
win = Series(tongji)[[2,3]]
new_win = Series(win,index=range(6))
new_lost.plot(kind='bar',figsize=(15,7),color='green')
new_win.plot(kind='bar',figsize=(15,7),color='red')

list1 = [u'第%s次交易'%(x+1) for x in range(int(len(myPrice)/2))]
list2 = data[['tradeTime']].values[::2]
plt.xticks(range(int(len(myPrice)/2)),[(x,y) for x,y in zip(list1,list2)],rotation=30)
#plt.xtiklabels(data[['tradeTime']][::2])
plt.ylim(-10,10)
plt.title(u'交易胜率统计')
#plt.savefig('tradetongji.png')
#data[['tradeTime']]
plt.show()

##############################读取数据库到csv###########################
# encoding: UTF-8


import pymongo
import pandas as pd
from pymongo import MongoClient
from pandas import Series,DataFrame
client = MongoClient('localhost', 27017)


# 从MongoDB中读取分钟数据，并将其转换为DataFrame格式。
db_min = client['VnTrader_1Min_Db'].IF1610# 选择合约分钟数据库
data_df = DataFrame()
datestr = "20161022" # 选择具体日期。
for dataset in db_min.find({'date':datestr}): # 按具体日期查找数据
    dataset_x = {'datetime_stp':Series(dataset['datetime']),'Close':Series(dataset['close']),
                'Open':Series(dataset['open']),'date':Series(dataset['time'][:8]),
                'Volume':Series(dataset['volume']),'High':Series(dataset['high']),
                'Low':Series(dataset['low'])}
    dataset_x_df = DataFrame(dataset_x)
    data_df = pd.concat([data_df, dataset_x_df], axis=0)
data_df.to_csv('MINU_IF1610_'+ datestr +'.csv',sep=',')
# print type(data_df.iloc[0]['datetime_stp'])
data_df


##############################k-means###########################
from numpy import *  
import random as rm
import matplotlib.pyplot as plt  

#%matplotlib inline

# calculate Euclidean distance  
def euclDistance(vector1, vector2):  
    return sqrt(sum(power(vector2 - vector1, 2)))  

# init centroids with random samples  
def initCentroids(dataSet, k):  
    numSamples, dim = dataSet.shape 
    centroids = dataSet[rm.sample(range(numSamples),k)]       
    return centroids  

# k-means cluster  
def kmeans(dataSet, k):  
    numSamples = dataSet.shape[0]  
    clusterAssment = zeros((numSamples, 2))  
    clusterChanged = True  

    ## step 1: init centroids  
    centroids = initCentroids(dataSet, k)  

    while clusterChanged:  
        clusterChanged = False  
        ## for each sample  
        for i in xrange(numSamples):  
            minDist  = 100000.0  
            minIndex = 0  
            ## for each centroid  
            ## step 2: find the centroid who is closest  
            for j in range(k):  
                distance = euclDistance(centroids[j], dataSet[i])  
                if distance < minDist:  
                    minDist  = distance  
                    minIndex = j  

            ## step 3: update its cluster  
            if clusterAssment[i, 0] != minIndex:  
                clusterChanged = True  
                clusterAssment[i] = minIndex#, minDist**2   

        ## step 4: update centroids  
        for j in range(k):  
            #pointsInCluster = dataSet[nonzero(clusterAssment[:, 0]== j)[0]] 
            pointsInCluster = dataSet[clusterAssment[:,0]== j] 
            centroids[j, :] = mean(pointsInCluster, axis = 0)  

    print 'Congratulations, cluster complete!'  
    return centroids, clusterAssment  

# show your cluster only available with 2-D data  
def showCluster(dataSet, k, centroids, clusterAssment):  
    numSamples, dim = dataSet.shape  
    if dim != 2:  
        print "dimension 2 please"  
        return 1  

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
    if k > len(mark):  
        print "k is too large!"  
        return 1  

    # draw all samples  
    for i in xrange(numSamples):  
        markIndex = int(clusterAssment[i, 0])  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  

    mark = ['Dr','Db','Dg','Dk','^r','+r','sr','dr','<r','pr']  
    # draw the centroids  
    for i in range(k):  
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12) 

    plt.title('k = %s' % k) 
    plt.show() 

#coding: utf-8

from numpy import *   
import matplotlib.pyplot as plt 
import re
from kmeans import *  

## step 1: load data  
print "step 1: load data..."  
dataSet = []  
fileIn = open('testSet.txt','r')  
for line in fileIn.readlines():  
    lineArr = line.strip()
    lineArr = re.split('\s+',lineArr)
    
    dataSet.append([float(lineArr[0]), float(lineArr[1])])  
  
## step 2: clustering...  
print "step 2: clustering..."  
dataSet = array(dataSet)  
k = input('enter a number: ')
centroids, clusterAssment = kmeans(dataSet, k)  
  
## step 3: show the result  
print "step 3: show the result..."  
showCluster(dataSet, k, centroids, clusterAssment)  




##############################最小二乘法###########################
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from datetime import datetime
import time
import scipy as sp  
from scipy.optimize import leastsq #最小二乘函数
import pylab as pl

%bookmark db C:/Users/Administrator/Desktop/data160627
%cd db
regularization =0.0000  # 正则化系数lambda  
m = 7   #多项式的次数
charge = 0.4/10000 #交易手续费
mainif_mounth = 9
mainif_day = 15
cal_start_hour = 9
cal_start_min = 30
analytime_hour = 10
analytime_min = 15

#多项式求导
def diff_poly(coe_list):
    temp = np.arange(len(coe_list))
    result=coe_list*temp[::-1]
    return result[:-1]
#拟合函数
def fake_func(p, x):
    f = np.poly1d(p) #产生以p列表为参数的多项式函数
    return f(x)     #返回计算代入值
#残差函数
def residuals(p, y, x):
    ret = y - fake_func(p, x)
    ret = np.append(ret, np.sqrt(regularization)*p) #将lambda^(1/2) * p加在了返回的array的后面
    return ret
#将每天分割开来
def judge_day_number(data): #data为时间序列
    timeslength = len(data)
    day={}
    for i in range(timeslength):
        daystr = data.index[i].strftime('%Y-%m-%d')
        if daystr in day:
            day[daystr] += 1
        else:
            day[daystr]=1
    day=Series(day)
    return day
# 选择最大合约的月份
def judge_day_mounth(data):#data为时间序列
    timeslength = len(data)
    day={}
    for i in range(timeslength):
        daymounth = data.index[i].month
        dayday = data.index[i].day
        if daymounth == mainif_mounth and dayday == mainif_day :
            day=data[i:]
            break
    return day 
# 导入分钟数据
def timestamp_datetime(value):
    format = '%H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt
# 判断一个时间是否属于某个具体的时间段
def choose_in_time(times): 
    timeslength = len(times)
    day={}
    for i in range(timeslength):
        if times.index[i].hour == cal_start_hour and times.index[i].minute >= cal_start_min:
            day = times[i:]
            break
    return day

def find_time(times): 
    timeslength = len(times)
    index = 0
    for i in range(timeslength):
        if times.index[i].hour == analytime_hour and times.index[i].minute == analytime_min :
            index = i
            break
    return index

def profitt(high,low):
    result = ( high * ( 1 - charge ) - low * ( 1 + charge) )/ (low * ( 1 + charge) )
    return result

data = pd.read_csv('MINU_IF1510.csv', header=0, sep=',') # 读取DAY_IF1506数据，为沪深300股指期货
# 将时间转换为标准格式时间
newdate  = pd.to_datetime(data['Time']) 
#newtime  = pd.to_datetime(data['time']) 
# 获取Time,LatestPrice,Volume指标
newdata = pd.concat([newdate,data[['Latestprice','Volume']]],axis = 1)
#转化为时间序列
newdata.index= newdata['Time']
del newdata['Time']
#选取最大合约日
day = judge_day_mounth(newdata)
#分割合约日天数
num = judge_day_number(day)
profit_total = 0
profit_if = 0
recall = {}
#按天提取
for i in range(len(num)):
    one_day = day[sum(num[:i]):sum(num[:i+1])]
    #选择回归开始时间后的数据
    one_day = choose_in_time( one_day )  
    day_num = len( one_day )
    date = str(one_day.index[0])[:10]
    print date
    one_day_time = Series(one_day.index,index=range(day_num))
    tempx=np.empty(day_num)
    tempx=range(day_num)
    tempy=np.array(one_day['Latestprice'])    
    #IF主力合约累计收益率
    profit_if += ( tempy[day_num - 1] * ( 1 - charge ) - tempy[0] * ( 1 + charge) )/ ( tempy[0] * ( 1 + charge) )
    singal = 0   #0是平仓 1是多仓 2是空仓
    correct = 0
    charge_times = 0
    pre_price = 0
    recall_temp = {} 
    recall_once = 0
    recall_pent = {}
    recall_index = 0
    analytime = find_time(one_day)
    for j in range(analytime,day_num):
        profit = 0
        x=tempx[:j]
        y=tempy[:j]
        #先随机产生一组多项式分布的参数
        Search_st = np.random.randn( m + 1 )
        plsq = leastsq(residuals, Search_st , args=(y, x))#第一个参数是需要拟合的差值函数，第二个是拟合初始值，第三个是传入函数的其他参数

        coe = plsq[0]
        diff1 = fake_func(diff_poly(coe),tempx[j])
        diff2 = fake_func(diff_poly(diff_poly(coe)),tempx[j])

        if diff1 * diff2 > 0  and j != day_num - 1:
            if singal == 0:
                pre_price = tempy[j]
                if diff1 > 0 :  
                    singal = 1
#                    print "开多" + "    " + "%.2f%%" %( profit_total * 100 )
                else:
                    singal = 2
#                    print "开空" + "    " + "%.2f%%" %( profit_total * 100 )
            elif singal !=0:
                recall_temp[j] = tempy[j]

        if diff1 * diff2 < 0 and singal != 0 and j != day_num - 1:
            charge_times += 1
            #计算最大回撤
            if recall_temp:
                recall_once = min ( recall_temp.values() )
                recall_pent[recall_index] = profitt (recall_once, pre_price)
            else:
                recall_pent[recall_index] = 0
            recall_temp = {}  
            recall_index += 1

            if singal == 1:    
                profit = profitt(tempy[j],pre_price)
            elif singal == 2:
                profit = profitt(pre_price,tempy[j])
            profit_total += profit
            if profit >= 0:
                correct += 1
            singal = 0
#            print "平仓" +  "    " + "%.2f%%" %( profit_total * 100 )

        elif j == day_num - 1 and singal != 0:
            charge_times += 1
            #计算最大回撤
            if recall_temp:
                recall_once = min ( recall_temp.values() )
                recall_pent[recall_index] = profitt (recall_once, pre_price)
            else:
                recall_pent[recall_index] = 0
            recall_temp = {}  
            recall_index += 1

            if singal == 1:    
                profit = profitt(tempy[j],pre_price)
            elif singal == 2:
                profit = profitt(pre_price,tempy[j])
            profit_total += profit
            if profit >= 0:
                correct += 1
            singal = 0
#            print "平仓" +  "    " + "%.2f%%" %( profit_total * 100 )

    recall_total = min (recall_pent.values())
    print "策略累计收益率：""%.2f%%" %( profit_total * 100 / len(num))
    print "IF主力合约累计收益率："+ "%.2f%%" %( profit_if * 100 ) 
    print "正确率："+ "%.2f%%" %( correct *100 / charge_times)
    print "最大回撤："+"%.2f%%" %( recall_total * 100 )
#    pl.plot(tempx, fake_func(plsq[0], tempx), label="%d"%(m)+" "'times fitted curve')
#    pl.plot(tempx,  tempy,  label='data points')
#    pl.legend()


##################################3min预测数据############################################
import pandas as pd 
import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from pandas import Series, DataFrame
from datetime import datetime
import random
#%matplotlib inline
def pre3min():
    plt.rcParams['font.sans-serif'] = ['SimHei']  #用来显示中文
    plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

    datestr = datetime.now().strftime('%Y%m%d%H%M%S')
    fig = plt.figure(1,figsize=(15,5))
    ax = fig.add_subplot(1,1,1)
    data3min1 = pd.read_csv('3min1.csv')
    data3min2 = pd.read_csv('3min2.csv')
    data3min = pd.concat([data3min1,data3min2])
    data2 = pd.read_csv('2.csv')
    #data1 = pd.read_csv('1.csv')
    record = []
    forindex = []
    panDuan = []
    for ti,panduan in zip(data3min.time,data3min['3minPre']):
        for da,index in zip(data2.date,data2.index):
            if ti[:5] == da[:5]:
                record.append(da)
                forindex.append(index)
                if 'up' in panduan:
                    panDuan.append(u'涨')
                else:
                    panDuan.append(u'跌')

    data = DataFrame({'record':record,'forindex':forindex,'panDuan':panDuan})
    data2.High.plot(ax = ax,linewidth=1.5)  
    for forindex,panduan,record in zip(data.forindex,data.panDuan,data.record):
        if panduan == u'跌':
            color = 'green'
        if panduan == u'涨':
            color = 'red'
        ax.annotate(panduan,color=color,xy=(forindex,data2.High[forindex]),fontsize=16)
    ax.set_title(u'交易涨跌预测\n放大到预测时间范围')
    ax.set_xlabel('datetime: %s'% datestr[:8])
    ax.set_ylabel('price')
    ax.set_xticks((data2.index.values[1::20]))
    ax.set_xticklabels(([date[:5] for date in ((data2['date'][1::20]).values)]), rotation=0)
    plt.subplots_adjust(hspace=0)
    ax.set_xlim([30,250])
    plt.savefig('%s.png'%datestr, bbox_inches='tight')
    plt.show()

pre3min()



###############################lbc##################################
# -*- coding: utf-8 -*-

#!!!!修改计算盈利亏损函数，根据实际交易价格而不是数据时间。

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def fwqdata_more_feature(data):
    ##0920不需要'time',应该是'data'
    try:
        data2 = data[['date','lastPrice','volume','openInterest','askPrice1','askVolume1','bidPrice1','bidVolume1']]
    except:
        data2 = data[['time','lastPrice','volume','openInterest','askPrice1','askVolume1','bidPrice1','bidVolume1']]
    data2.columns=['time','price','volume','position','askPrice1','askVolume1','bidPrice1','bidVolume1']
    #print(len(data2))
    timenew='09:50'
    maxpr=[]
    minpr=[]
    vol=[]
    close=[]
    pos=[]
    askpr1=[]
    askvo1=[]
    bidpr1=[] 
    bidvo1=[]

    maxpr1=0
    minpr1=999999
    ##0920数据不同，不需要这条判断。
    for i in range(len(data2)):
        data=data2.ix[i]
        if data['time'][:5] == timenew:
            begin = i
            break
    print begin
    
    data2 = data2[begin:]
    data2.index=range(len(data2))
    #print(len(data2))
    
    for i in range(len(data2)):
        data=data2.ix[i]
        if data['time'][:5] == timenew:
            maxpr1=max(maxpr1,data['price'])
            minpr1=min(minpr1,data['price'])
            ap1 = data['askPrice1']
            av1 = data['askVolume1']
            bp1 = data['bidPrice1']
            bv1 = data['bidVolume1']
        else:
            if maxpr1!=0:##每天最后几个数据有问题需要扔掉。
                maxpr.append(maxpr1)
                minpr.append(minpr1)
                askpr1.append(ap1)
                askvo1.append(av1)
                bidpr1.append(bp1)
                bidvo1.append(bv1)
            
                close.append(data['price'])
                maxpr1=0
                minpr1=99999
                timenew= data['time'][:5]
                vol_end = data['volume']
                pos_end = data['position']
                vol.append(vol_end)
                pos.append(pos_end)
            
    new_vol=np.array(vol)[1:]-np.array(vol)[:-1]##2min开头的结果减去1min开头的结果，得到1min的volume
    new_pos = np.array(pos)[1:]-np.array(pos)[:-1]
    new_max=np.array(maxpr)[1:]
    new_min=np.array(minpr)[1:]
    askpr1 = np.array(askpr1)[1:]
    askvo1 = np.array(askvo1)[1:]
    bidpr1 = np.array(bidpr1)[1:]
    bidvo1 = np.array(bidvo1)[1:]
    close = np.array(close)[1:]
    ratio_pr_vol = close/new_vol
    #rint len(close),len(askpr1),len(askvo1),len(bidpr1),len(bidvo1)
    Data = pd.DataFrame([new_max,new_min,ratio_pr_vol,new_vol,new_pos,askpr1,askvo1,bidpr1,bidvo1,close])
    Data = Data.T
    ratio_pr_vol = close/new_vol
    Data.columns = ['max','min','ratio_pr_vol','vol','pos','askpr1','askvo1','bidpr1','bidvo1','close']

    EMA_12 = pd.ewma(close, span=12)
    EMA_26 = pd.ewma(close, span=26)

    Data['DIF'] = EMA_12 - EMA_26
    Data['DEA'] = pd.ewma(Data['DIF'], 9)
    dif = Data['DIF'].values
    dea = Data['DEA'].values
    barh = []
    for i in range(len(dif)):
        barh.append(2*(dif[i]-dea[i]))
    Data['BAR'] = barh

    high = Data['max'].values
    low = Data['min'].values
    
               
    Data['mean']=(Data['max']+Data['min']+close)/3.0

    mtr = [0]
    for i in range(1,len(high)):
    
        a = high[i] - low[i]
        b = np.abs(high[i]-close[i-1])
        c = np.abs(close[i-1]-low[i])
        mtr.append(max(a,b,c))
    Data['MTR'] = mtr
    Data['ATR'] = pd.rolling_mean(Data['MTR'], 10)
    
    Data['KCM'] = pd.ewma(Data['mean'], span=20)
    kcm = Data['KCM'].values
    atr = Data['ATR'].values
    kcu = []
    kcb = []
    for t in range(len(kcm)):
        kcu.append(kcm[t]+2*atr[t])
        kcb.append(kcm[t]-2*atr[t])
    
    Data['KCU'] = kcu
    Data['KCB'] = kcb
    Data['MA_5'] = pd.rolling_mean(Data['close'], 5)
    Data['MA_10'] = pd.rolling_mean(Data['close'], 10)
    Data['MA_15'] = pd.rolling_mean(Data['close'], 15)
    print len(Data)
    Data = Data.dropna()
    print len(Data)
    return Data
    
    
    
def label_build(Data):
    label = []
    Data.index=range(len(Data))
    for i in range(len(Data)):
        data = Data.ix[i,:]
        if data['MA_5']<data['MA_10']<data['MA_15']:
            label.append(-1)
        elif data['MA_5']>data['MA_10']>data['MA_15']:
            label.append(1)
        else:
            label.append(0)
    return label

def error_compare(true, pred):
    up = 0
    down = 0
    error_up = 0
    error_down = 0
    for i in range(len(pred)):
        if pred[i] == 1:
            up+=1
            if true[i] != 1:error_up+=1
        if pred[i] == -1: 
            down+=1
            if true[i] != -1:error_down += 1
                
    error = float(error_up + error_down)/(up + down)
    return error

##调用该函数输出误差估计。
def error_print(files):
    dtnew = pd.read_csv(files)
    dtnew=fwqdata_more_feature(dtnew)
    model = pickle.load(file('xgb0921.pkl','r'))
    test_label= label_build(dtnew)
    test_pred = model.predict(dtnew[:-1])
    error = error_compare(test_label[1:], test_pred)
    return error

##为了绘图用所以保存时间。

def plot_time(data2):###change time here.
    try:
        data2 = data2[['date','lastPrice','volume','openInterest','askPrice1','askVolume1','bidPrice1','bidVolume1']]
    except:
        data2 = data2[['time','lastPrice','volume','openInterest','askPrice1','askVolume1','bidPrice1','bidVolume1']]
    data2.columns=['time','price','volume','position','askPrice1','askVolume1','bidPrice1','bidVolume1']
    timenew='09:45'
    maxpr=[]
    
    close=[]
    
    times = []
    
    maxpr1=0
    
    for i in range(len(data2)):
        data=data2.ix[i]
        if data['time'][:5] == timenew:
            begin = i
            break
    print begin
    
    data2 = data2[begin:]
    data2.index=range(len(data2))
    
    for i in range(len(data2)):
        data=data2.ix[i]
        if data['time'][:5] == timenew:
            maxpr1=max(maxpr1,data['price'])
            
        else:
            if maxpr1!=0:##每天最后几个数据有问题需要扔掉。
                
                
                close.append(data['price'])
                
                timenew= data['time'][:5]
                
                times.append(timenew)
    
    Data = pd.DataFrame([np.array(close),np.array(times)], index = ['close', 'time'])
    return Data

def plot_ans(duolist, pingduo_list, konglist, pingkong_list, price, num=0):
    if len(duolist)==0 and len(konglist)==0: 
        #print u'non'
        return 
    elif len(duolist)==0 and len(konglist)>0:
        a, b, konglist, pingkong_list = droplist(konglist, pingkong_list)
        k = ['short', 'cover']
        #print 1
    elif len(konglist)==0 and len(duolist)>0:
        a, b, duolist, pingduo_list = droplist(duolist, pingduo_list)
        k = ['buy', 'sell']
        #print 2
    elif duolist[0]<konglist[0]:
        a, b, duolist, pingduo_list = droplist(duolist, pingduo_list)
        k = ['buy', 'sell']
    else:
        a, b, konglist, pingkong_list = droplist(konglist, pingkong_list)
        k = ['short', 'cover']
    i=a[0]
    j=b[0]
    color = ['red', 'green']
    col = color[num%2]
    plot_ij(price,i,j,k, col)
    plot_ans(duolist, pingduo_list, konglist, pingkong_list, price, num=num+1)
    
    
def droplist(list1, list2):
    a = list1
    b = list2
    
    list1 = list1.drop(list1[0])
    list2 = list2.drop(list2[0])
    return a, b, list1, list2

def plot_ij(price,i,j,k,col):
    plt.annotate(u'%s'%(k[0]),xy=(i,price[i]),xytext=(i+0.2,price[i]+0.2),fontsize=20,arrowprops=dict(facecolor=col, shrink=0.05))
    plt.annotate(u'%s'%(k[1]),xy=(j,price[j]),xytext=(j+0.2,price[j]+0.2),fontsize=20,arrowprops=dict(facecolor=col, shrink=0.05))

def trade_list(data, duo, pingduo, kong, pingkong):
    duodata = data[data['time'].map(lambda x:(str(x)[:5] in duo))]
    pingduo_data = data[data['time'].map(lambda x:(str(x)[:5] in pingduo))]
    kongdata = data[data['time'].map(lambda x:(str(x)[:5] in kong))]
    pingkong_data = data[data['time'].map(lambda x:(str(x)[:5] in pingkong))]

    duolist=duodata.index
    pingduo_list = pingduo_data.index
    konglist = kongdata.index
    pingkong_list = pingkong_data.index 
    return duolist, pingduo_list, konglist, pingkong_list
    
    
##输出收益率。    
def rate_count(duo_pr, pingduo_pr, kong_pr, pingkong_pr ):
    rate=[]
    win_time=0
    for m in range(len(duo_pr)):
        pc = pingduo_pr[m]
        jc = duo_pr[m]
        #return pc
        print pc
        print jc
        if pc>jc:
            win_time+=1
            print(m,u'duo',win_time)
        rate.append(pc/jc - 1)
    
    for m in range(len(kong_pr)):
        pc = pingkong_pr[m]
        jc= kong_pr[m]
        print pc
        print jc
        if pc<jc:
            win_time+=1
            #print(m,u'kong',win_time)
        rate.append(1 - pc/jc)
    return np.sum(rate), np.sum(rate)/len(rate), float(win_time)/len(rate)

##绘图。
def plot_trade(price, duolist, pingduo_list, konglist, pingkong_list):
    fig=plt.figure(figsize=(16,8))
    ax=plt.subplot(111)
    
    plt.plot(price)
    
    plot_ans(duolist, pingduo_list, konglist, pingkong_list, price)
    
##调用该函数输出误差并绘图
def all_in_all(file_name, duo, pingduo, kong, pingkong, file_trade_name):    
    dtnew = pd.read_csv(file_name)
    trade = pd.read_csv(file_trade_name)
    #trade = file_trade_name
    data = plot_time(dtnew)
    data = data.T
    price = data['close']
    
    duo, pingduo, kong, pingkong, duo_pr, pingduo_pr, kong_pr, pingkong_pr = time_print(duo, pingduo, kong, pingkong, trade)
    a, b, c = rate_count(duo_pr, pingduo_pr, kong_pr, pingkong_pr)
    print(a, b,c)
    duolist, pingduo_list, konglist, pingkong_list = trade_list(data, duo, pingduo, kong, pingkong)
    plot_trade(price, duolist, pingduo_list, konglist, pingkong_list)
    
###根据交易情况给出实时的价格时间和交易价格
def trade_pr_time(id_list,trade):
    try:
        list1 = trade[trade['orderID'].map(lambda x :x in id_list)]
    except:
        list1 = trade[trade['orderID'].map(lambda x :x in id_list)]
    list2 = list1['price']
    
    list1 = list1['orderTime']##这里提供的是委托部分，而不是交易部分的价格。
    list1 = list(list1)
    list2 = list(list2)
    a = []
    for k in list1:
        k = k[:5]
        a.append(k)
    return a,list2

def time_print(duo, pingduo, kong, pingkong, trade):##修改调用该函数所用到的函数。返回的是时间和交易价格的list
    duo, duo_pr = trade_pr_time(duo, trade)
    pingduo, pingduo_pr = trade_pr_time(pingduo, trade)
    kong, kong_pr = trade_pr_time(kong, trade)
    pingkong, pingkong_pr = trade_pr_time(pingkong, trade)
    return duo, pingduo, kong, pingkong, duo_pr, pingduo_pr, kong_pr, pingkong_pr







#############################lbc2###########################################
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
#import xgboost as xgb
import pickle
import matplotlib.pyplot as plt


def file_names_get(file_dir='csv'):##从文件夹中获取所有文件名。
    import os
    file_names = []
    for root, dirs, files in os.walk(file_dir):  
        print(root) #当前目录路径  
        print(dirs) #当前路径下所有子目录  
        print(files) #当前路径下所有非目录子文件 
        file_names.append(files)
    return file_names


def hist_more_feature(data):##从历史数据中整理得到所需要的分钟格式的dataframe 
    dt2 =data[[u'MaxPrice',u'MinPrice',  u'Volume',u'delta_Position', u'Last.Sell1price',u'Last.Sell1quantity',
         u'Last.Buy1price',u'Last.Buy1quantity', u'Latestprice',u'DIF', u'DEA', u'BAR',u'MTR',
         u'ATR', u'KCM', u'KCU', u'KCB']]
    dt2.columns=[u'max', u'min', u'vol', u'pos', u'askpr1', u'askvo1',
       u'bidpr1', u'bidvo1', u'close', u'DIF', u'DEA', u'BAR', u'MTR',
       u'ATR', u'KCM', u'KCU', u'KCB']
    dt2['mean']=(dt2['max']+dt2['min']+dt2['close'])/3
    ratio = dt2['close']/dt2['vol']
    for i in range(len(ratio)):
        if dt2['vol'][i]==0:
            ratio[i]=0
    dt2['ratio_pr_vol'] = ratio
    dt2['MA_5'] = pd.rolling_mean(dt2['close'], 5)
    dt2['MA_10'] = pd.rolling_mean(dt2['close'], 10)
    dt2['MA_15'] = pd.rolling_mean(dt2['close'], 15)
    dt3 = dt2[[u'max', u'min', u'ratio_pr_vol', u'vol', u'pos', u'askpr1', u'askvo1',
       u'bidpr1', u'bidvo1', u'close', u'DIF', u'DEA', u'BAR', u'mean', u'MTR',
       u'ATR', u'KCM', u'KCU', u'KCB', u'MA_5', u'MA_10', u'MA_15']] 
    dt3 = dt3.dropna()
    return dt3
    

def lstm_data_print(file_names, n):##对于一系列文件名，分别得到样本和标签；n为时间序列的长度。
    datas = []
    labels = []
    for file_name in file_names[0]:
        data, label = lstm_data_prep(file_name, n)
        datas.append(data)
        labels.append(label)
    newdatas = []
    for data in datas:
        for little in data:
            newdatas.append(little)
    newlabels = []
    for label in labels:
        for single in label:
            newlabels.append(single)
    return newdatas, newlabels

def lstm_data_prep(file_name, n):
    t1 = pd.read_csv(file_name)
    if file_name[:2]=='dt':
        dt1 = fwqdata_more_feature(t1)
    else:
        dt1 = hist_more_feature(t1)
    ls_datas = []
    npdt1 = np.array(dt1)
    for i in range(len(npdt1)-n):
        lsdt1 = npdt1[i:i+n]
        ls_datas.append(lsdt1)
    la1 = label_build(dt1)
    la1 = la1[n:]##这里的label是当前时刻的，因此作为预测要往后挪一个，即数据是0-9，label为1-10
    ls_labels=[]
    for i in la1:
        if i ==1:ls_label=[1,0,0]
        if i ==0:ls_label=[0,1,0]
        if i ==-1:ls_label=[0,0,1]
        ls_labels.append(ls_label)
    return ls_datas,ls_labels

def label_build(Data):#根据处理后的data得到对应的标签。
    label = []
    Data.index=range(len(Data))
    for i in range(len(Data)):
        data = Data.ix[i,:]
        if data['MA_5']<data['MA_10']<data['MA_15']:
            label.append(-1)
        elif data['MA_5']>data['MA_10']>data['MA_15']:
            label.append(1)
        else:
            label.append(0)
    print(len(label),u'lenth of label')
    return label

def fwqdata_more_feature(data):##从服务器上爬得数据处理成分钟格式的dataframe
    ##0920不需要'time',应该是'data'
    if 'date' in data.columns:
        data2 = data[['date','lastPrice','volume','openInterest','askPrice1','askVolume1','bidPrice1','bidVolume1']]
    else:
        data2 = data[['time','lastPrice','volume','openInterest','askPrice1','askVolume1','bidPrice1','bidVolume1']]
    
    data2.columns=['time','price','volume','position','askPrice1','askVolume1','bidPrice1','bidVolume1']
    #print(len(data2))
    timenew='09:36'
    maxpr=[]
    minpr=[]
    vol=[]
    close=[]
    pos=[]
    askpr1=[]
    askvo1=[]
    bidpr1=[] 
    bidvo1=[]

    maxpr1=0
    minpr1=999999
    ##0920数据不同，所有不需要这条判断。
    for i in range(len(data2)):
        data=data2.ix[i]
        if data['time'][:5] == timenew:
            begin = i
            break
    #print begin
    
    data2 = data2[begin:]
    data2.index=range(len(data2))
    #print(len(data2))
    
    for i in range(len(data2)):
        data=data2.ix[i]
        if data['time'][:5] == timenew:
            maxpr1=max(maxpr1,data['price'])
            minpr1=min(minpr1,data['price'])
            ap1 = data['askPrice1']
            av1 = data['askVolume1']
            bp1 = data['bidPrice1']
            bv1 = data['bidVolume1']
        else:
            if maxpr1!=0:##每天最后几个数据有问题需要扔掉。
                maxpr.append(maxpr1)
                minpr.append(minpr1)
                askpr1.append(ap1)
                askvo1.append(av1)
                bidpr1.append(bp1)
                bidvo1.append(bv1)
            
                close.append(data['price'])
                maxpr1=0
                minpr1=99999
                timenew= data['time'][:5]
                vol_end = data['volume']
                pos_end = data['position']
                vol.append(vol_end)
                pos.append(pos_end)
            
    new_vol=np.array(vol)[1:]-np.array(vol)[:-1]##2min开头的结果减去1min开头的结果，得到1min的volume
    new_pos = np.array(pos)[1:]-np.array(pos)[:-1]
    new_max=np.array(maxpr)[1:]
    new_min=np.array(minpr)[1:]
    askpr1 = np.array(askpr1)[1:]
    askvo1 = np.array(askvo1)[1:]
    bidpr1 = np.array(bidpr1)[1:]
    bidvo1 = np.array(bidvo1)[1:]
    close = np.array(close)[1:]
    ratio_pr_vol = close/new_vol
    for i in range(len(close)):
        if new_vol[i] == 0: 
            print i
            ratio_pr_vol[i] = 0
    #rint len(close),len(askpr1),len(askvo1),len(bidpr1),len(bidvo1)
    Data = pd.DataFrame([new_max,new_min,ratio_pr_vol,new_vol,new_pos,askpr1,askvo1,bidpr1,bidvo1,close])
    Data = Data.T
    ratio_pr_vol = close/new_vol
    Data.columns = ['max','min','ratio_pr_vol','vol','pos','askpr1','askvo1','bidpr1','bidvo1','close']

    EMA_12 = pd.ewma(close, span=12)
    EMA_26 = pd.ewma(close, span=26)

    Data['DIF'] = EMA_12 - EMA_26
    Data['DEA'] = pd.ewma(Data['DIF'], 9)
    dif = Data['DIF'].values
    dea = Data['DEA'].values
    barh = []
    for i in range(len(dif)):
        barh.append(2*(dif[i]-dea[i]))
    Data['BAR'] = barh

    high = Data['max'].values
    low = Data['min'].values
    
               
    Data['mean']=(Data['max']+Data['min']+close)/3.0

    mtr = [0]
    for i in range(1,len(high)):
    
        a = high[i] - low[i]
        b = np.abs(high[i]-close[i-1])
        c = np.abs(close[i-1]-low[i])
        mtr.append(max(a,b,c))
    Data['MTR'] = mtr
    Data['ATR'] = pd.rolling_mean(Data['MTR'], 10)
    Data['KCM'] = pd.ewma(Data['mean'], span=20)
    kcm = Data['KCM'].values
    atr = Data['ATR'].values
    kcu = []
    kcb = []
    for t in range(len(kcm)):
        kcu.append(kcm[t]+2*atr[t])
        kcb.append(kcm[t]-2*atr[t])
    
    Data['KCU'] = kcu
    Data['KCB'] = kcb
    Data['MA_5'] = pd.rolling_mean(Data['close'], 5)
    Data['MA_10'] = pd.rolling_mean(Data['close'], 10)
    Data['MA_15'] = pd.rolling_mean(Data['close'], 15)
    print len(Data)
    Data = Data.dropna()
    print len(Data),u'lenth of data'
    return Data










########################ARIMA###########################################
# coding:utf-8


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 

from statsmodels.graphics.api import qqplot  #用来作QQ图
from statsmodels.tsa.stattools import adfuller #ADF单位根检验
from statsmodels.tsa.stattools import acf,pacf #自相关，偏自相关
from statsmodels.tsa.arima_model import ARMA  #ARIMA处理
from statsmodels.tsa.seasonal import seasonal_decompose #用来季节性分解处理
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf#用来做自相关，偏自相关图
from scipy import stats
from pandas import Series,DataFrame


class ARMAModelTest:
    '''
    基于ARIMA模型，对时间序列做平稳化处理，利用差分查找最佳d值，
    根据平稳化处理后的时间序列自相关、偏自相关图选择合适的p,q值，
    对ARMA模型进行AIC、BIC、HQIC准则验证，
    检查残差序列：分别用自相关、偏自相关图，D-W检验，正态分布检验，qq图检验，Ljung-Box
    检验，
    对接下来的走势进行预测。
    '''

    def __init__(self,time,num=100,lags=40):
        self.time = time
        self.num = num
        self.lags = lags

    #第一步：对时间序列做平稳化处理（一般使用差分）
    def ser_stable_plot(self,timeSer,target='Latestprice',diff=1):
        global data 
        data = timeSer.ix[self.time][target][:self.num].diff(diff)
        data.index = pd.date_range(data.index[0],periods=len(data))
        data.plot(use_index=False,figsize=(12,6),linewidth=2,grid=True)
        plt.show()
        #print data

    #第二步：选择合适的p,q
    def p_q_choice(self,timeSer):
        #fig1 = sm.graphics.tsa.plot_acf(data,lags=lags,ax=ax[0])
        #fig2 = sm.graphics.tsa.plot_pacf(data,lags=lags,ax=ax[1])
        timeSer = timeSer.ix[self.time]
        timeSer.index = pd.date_range(timeSer.index[0],periods=len(timeSer))
        figure001 = plt.figure(figsize=(12,6))  
        ax1 = figure001.add_subplot(211)
        ax2 = figure001.add_subplot(212)    
        fig1 = plot_acf(timeSer[:self.num],lags=self.lags,ax=ax1)
        fig2 = plot_pacf(timeSer[:self.num],lags=self.lags,ax=ax2)
        plt.show()  

    #第三步：采用ARMA模型的AIC法则
    def arma_mod(self,p,q,timeSer):
        #arma_mod = sm.tsa.ARMA(data,order=p_q_choice(lags)).fit()
        global ar_mod
        global ma_mod 
        global arma_mod

        timeSer = timeSer.ix[self.time]
        timeSer.index = pd.date_range(timeSer.index[0],periods=len(timeSer)) 

        ar_mod = ARMA(timeSer[:self.num],order=(p,0)).fit()
        print u'-----------AR模型-----------'
        print 'AIC: ',ar_mod.aic 
        print 'BIC: ',ar_mod.bic 
        print 'HQIC: ',ar_mod.hqic 
        ma_mod = ARMA(timeSer[:self.num],order=(0,q)).fit()
        print u'-----------MA模型-----------'
        print 'AIC: ',ma_mod.aic 
        print 'BIC: ',ma_mod.bic 
        print 'HQIC: ',ma_mod.hqic 
        arma_mod = ARMA(timeSer[:self.num],order=(p,q)).fit()
        print u'-----------ARMA模型-----------'
        print 'AIC: ',arma_mod.aic 
        print 'BIC: ',arma_mod.bic 
        print 'HQIC: ',arma_mod.hqic 

    #第四步：ADF单位根检验，检查残差序列，D-W（德宾-沃森)检验
    def check_rasid_DW(self):
        print 'check_rasid_DW'

        ar_resid = adfuller(ar_mod.resid)
        ma_resid = adfuller(ma_mod.resid)
        arma_resid = adfuller(arma_mod.resid)

        df1 = Series(ar_resid[0:4],index=['Test Statistic','p-value','Lags Used','Nob'])
        for key, value in ar_resid[4].items():
            df1['Critical Value(%s)'%key] = value 

        df2 = Series(ma_resid[0:4],index=['Test Statistic','p-value','Lags Used','Nob'])
        for key,value in ma_resid[4].items():
            df2['Critical Value(%s)'%key] = value 

        df3 = Series(arma_resid[0:4],index=['Test Statistic','p-value','Lags Used','Nob'])
        for key,value in arma_resid[4].items():
            df3['Critical Value(%s)'%key] = value 

        df = pd.concat([df1,df2,df3],axis=1,keys=['ar_resid','ma_resid','arma_resid'])
        print df 

        figure002 = plt.figure(figsize=(12,6))  
        ax3 = figure002.add_subplot(211)
        ax4 = figure002.add_subplot(212)
        fig3 = plot_acf(arma_mod.resid,lags=self.lags,ax=ax3)
        fig4 = plot_pacf(arma_mod.resid,lags=self.lags,ax=ax4)
        plt.show()
        #第五步：D-W（德宾-沃森)检验
        dw = sm.stats.durbin_watson(arma_mod.resid.values)
        print 'D-W: ',dw


    #第六步：对残差做正态分布检验
    def check_norm_qq(self,):
        norm = stats.normaltest(arma_mod.resid)
        print norm 

        figure003 = plt.figure(figsize=(12,6))  
        ax5 = figure003.add_subplot(111)        
        figqq = qqplot(arma_mod.resid,ax=ax5,fit=True,line='q')
        plt.show()

    #第七步：残差Ljung-Box检验（Q检验）
    def check_q(self,):
        r,q,p = sm.tsa.acf(arma_mod.resid,qstat=True)
        df = np.c_[range(1,41),r[1:],q,p]
        frame = DataFrame(df,columns=['Lags','AC','Q','Prob(>Q)'])
        frame.set_index('Lags')
        pct = len(frame[frame['Prob(>Q)']>0.05])/len(frame)
        print 'pct',pct

    #第八步：预测
    def predict_target(self,timeSer,start=None,end=None,dynamic=False):
        #timeSer.index = pd.date_range(timeSer.index[0],periods=len(timeSer))
        timeSer = timeSer.ix[self.time]
        timeSer_handle = timeSer.ix[:self.num]
        timeSer_handle.index = pd.date_range(timeSer_handle.index[0],periods=len(timeSer_handle))
        #print timeSer_handle
        predict_target = arma_mod.predict(start=start,end=end,dynamic=dynamic)
        print predict_target 
        figure004 = plt.figure(figsize=(12,6))  
        ax6 = figure004.add_subplot(111)        
        timeSer_handle.plot(ax=ax6,lw=2)        
        arma_mod.plot_predict(start,end,dynamic=dynamic,ax=ax6,plot_insample=False)
        #return figs1,figs2
        plt.show()
        

if __name__ == '__main__':
    dataNeed = pd.read_csv('dataFortime.csv')
    dataNeed2 = dataNeed.ix[:,1:3].set_index('Time')
    dataNeed3 = dataNeed.ix[:,1:3].set_index('Time')
    dataNeed3.index = pd.to_datetime(dataNeed3.index)
    dataNeed3.index = dataNeed3.index.map(lambda x:x.strftime('%Y-%m-%d'))
    dataNeed3.index = pd.to_datetime(dataNeed3.index)

    dataNeed2.index = pd.to_datetime(dataNeed2.index)
    dataNeed2.index = dataNeed2.index.map(lambda x:x.strftime('%Y-%m-%d'))
    dataNeed2.index = pd.to_datetime(dataNeed2.index)
    timeRight = dataNeed2.index.unique()
    print u'有记录日期为： ',timeRight

    time = raw_input(u'enter a date for test: ')
    while time not in timeRight:
        print u'你输入的时间不在记录日期内，请重新输入!'
        time = raw_input(u'enter a date for test: ')

    test = ARMAModelTest(time=time,num=100,lags=40)
    test.ser_stable_plot(dataNeed2,target='Latestprice',diff=1)
    test.p_q_choice(dataNeed2)
    p = input(u'enter num p: ')
    q = input(u'enter num q: ')
    test.arma_mod(p,q,dataNeed2)
    test.check_rasid_DW()
    test.check_norm_qq()
    test.check_q()

    dataNeed3 = dataNeed3.ix[time][:100]
    dataNeed3.index = pd.date_range((str(dataNeed3.index[0])[:10]),periods=len(dataNeed3))

    print u'real trend end at',str(dataNeed3.index[99])

    end = False
    while not end:
        try:
            start = raw_input(u'enter predict when to begin: ')
            end = raw_input(u'enter predict when to end: ')
        except:
            print u'数据有误，请确认后重新输入！'
        certain = raw_input(u'请输入是否加入已有数据的验证：（yes/no)')
        if certain.strip() == 'yes':
            dynamic = False
            print 'yes'
        else:
            dynamic = True 
            print 'no'
        test.predict_target(dataNeed2,start=start,end=end,dynamic=dynamic)
        panduan = raw_input(u'是否结束预测：（yes/no)')
        if panduan.strip() == 'yes':
            end = True
        else:
            end = False






##############################################################################

from matplotlib import pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from datetime import datetime
%matplotlib inline 

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def text_to_df(path):
    import codecs
    from pandas import Series
    List = []
    Lindex = []
    Lvalues = []
    with codecs.open(path,'r','gbk') as f:
        data = f.readlines()
    for line in data:
        if 'predict_value' in line:
            List.append(line)    
    for L in List:
        Lindex.append(L[:8])
        Lvalues.append(L.split('predict_value')[1].strip())
    ser = Series(Lvalues,index=Lindex)
    return ser.map(lambda x:float(x))

#ser = text_to_df(path)

def join_df(path1,path2,dateNeed=(datetime.now().strftime('%Y%m%d%H%M%S'))):
    data= pd.read_csv(path1)
    data.index = data.date
    data = data[['Open']]  
    
    ser = text_to_df(path2)
    ser.index = ser.index.map(lambda x:x[:5])
    data.index = data.index.map(lambda x:x[:5])
    serDF = DataFrame(ser)
    df = serDF.join(data,how='right')
    
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    plt.title(u'Date：%s'%dateNeed)
    plt.fill_between(range(len(df)),df.ix[:,0],color='red',alpha=.5)
    plt.axhline(y=0,ls='--',lw=2,color='gray')
    plt.axhline(y=1.8,ls='--',lw=2,color='green')
    plt.axhline(y=-2,ls='--',lw=2,color='red')
    plt.text(10,1.8,u'理论建空仓分界线',fontdict=dict(color='purple',size=12))
    plt.text(10,-2,u'理论建多仓分界线',fontdict=dict(color='purple',size=12))
    plt.ylabel(u'p-value输出')
    plt.xlabel(u'时间（Time）')
    df.Open.plot(ax=ax.twinx())
    plt.ylabel(u'价格走势');
    plt.savefig('D:\work store private\pic_set\%s.png'%dateNeed)

#def plot_show(path1,path2):
#    ser = text_to_df(path1)
#    join_df(path2)

##############################################################################

##############################################################################

##############################################################################

##############################################################################

##############################################################################

##############################################################################
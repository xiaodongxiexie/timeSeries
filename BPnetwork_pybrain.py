# -*- coding: utf-8 -*-
# @Author: xiaodong
# @Date:   2017-04-27 00:44:26
# @Last Modified by:   xiaodong
# @Last Modified time: 2017-04-27 00:44:33
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, TanhLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

#从sklearn数据集中读取用来模拟的数据
boston = load_boston()
x = boston.data
y = boston.target.reshape(-1,1)

#直接采用不打乱的方式进行7:3分离训练集和测试集
per = int(len(x) * 0.7)

#对数据进行归一化处理（一般来说使用Sigmoid时一定要归一化）
sx = MinMaxScaler()
sy = MinMaxScaler()
xTrain = x[:per]
xTrain = sx.fit_transform(xTrain)
yTrain = y[:per]
yTrain = sy.fit_transform(yTrain)

xTest = x[per:]
xTest = sx.transform(xTest)
yTest = y[per:]
yTest = sy.transform(yTest)

#初始化前馈神经网络
fnn = FeedForwardNetwork()

#构建输入层，隐藏层和输出层，一般隐藏层为3-5层，不宜过多
inLayer = LinearLayer(x.shape[1], 'inLayer')
hiddenLayer = TanhLayer(3, 'hiddenLayer')
outLayer = LinearLayer(1, 'outLayer')

#将构建的输出层、隐藏层、输出层加入到fnn中
fnn.addInputModule(inLayer)
fnn.addModule(hiddenLayer)
fnn.addOutputModule(outLayer)

#对各层之间建立完全连接
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

#与fnn建立连接
fnn.addConnection(in_to_hidden)
fnn.addConnection(hidden_to_out)
fnn.sortModules()

#初始化监督数据集
DS = SupervisedDataSet(x.shape[1],1)

#将训练的数据及标签加入到DS中
for i in range(len(xTrain)):
    DS.addSample(xTrain[i],yTrain[i])

#采用BP进行训练，训练至收敛，最大训练次数为1000
trainer = BackpropTrainer(fnn, DS, learningrate=0.01, verbose=True)
trainer.trainUntilConvergence(maxEpochs=1000)


#在测试集上对其效果做验证
values = []
for x in xTest:
    values.append(sy.inverse_transform(fnn.activate(x))[0])

#计算RMSE (Root Mean Squared Error)均方差
sum(map(lambda x: x ** 0.5,map(lambda x,y: pow(x-y,2), boston.target[per:], values))) / float(len(xTest))

#将训练数据进行保存
NetworkWriter.writeToFile(fnn, 'pathName.xml')
joblib.dump(sx, 'sx.pkl', compress=3)
joblib.dump(sy, 'sy.pkl', compress=3)

#将保存的数据读取
fnn = NetworkReader.readFrom('pathName.xml')
sx = joblib.load('sx.pkl')
sy = joblib.load('sy.pkl')
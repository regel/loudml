#!/usr/bin/env python

import json
import numpy as np
import matplotlib.pyplot as plt
#import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
np.random.seed(7)
# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
anomaly_threshold = 30.0
look_back = 5
# dlen = 24 hours = 24*60 data points used in training
dlen=60*24
m = np.zeros((dlen, 1), dtype=float)

def walk(d):
    t0=0
    for k in d['aggregations']['per_minute']['buckets']:
        timestamp=k['key']
        timeval=k['key_as_string']
        val=k['avg_in_mos']['value']
        if val != None:
            if t0 == 0:
                yield 0, val, timeval
                t0=timestamp
            else:
                yield (timestamp-t0)/1000, val, timeval

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


with open('input.json') as json_data:
    j=0
    d = json.load(json_data)
    for key, val, timeval in walk(d):
        m[j] = [val]
        j = j+1
        if j == dlen:
            break


scaler = MaxAbsScaler()
#scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(m)

#print (dataset)

#print m
#x = m[:,0]
#y = dataset
#plt.plot(x, y)
#plt.show()
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#print(len(train), len(test))

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# input length to model.predict is loop_back long
#print(testX[10:20], model.predict(testX[10:20]))

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

with open('input.json') as json_data:
    j = 0
    X = np.zeros((1,look_back), dtype=float)
    d = json.load(json_data)
    for _, val, timeval in walk(d):
        j = j+1
        X=np.roll(X,-1,axis=1)
        X[0][look_back-1] = val / 5.0
        if j < look_back:
            continue
        X_ = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        Y_ = model.predict(X_, batch_size=1, verbose=0)
        score = abs(Y_[0][0] - X[0][look_back-1]) * 100
        if score > anomaly_threshold:
            print("Anomaly @timestamp:", timeval, X_, Y_, score)





#!/usr/bin/env python

import json
import numpy as np
import matplotlib.pyplot as plt
#import pandas
import math
# fix random seed for reproducibility.
# WARN: Open issue: Keras API reproducibility #11585
# https://github.com/tensorflow/tensorflow/issues/11585
np.random.seed(7)
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.layers import LSTM
from tensorflow.contrib.keras.api.keras.callbacks import TensorBoard

#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
np.random.seed(7)
# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
num_features = 1
look_back = 5
batch_size = 32
hidden_neurons=4
num_epochs=10
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

def _load_data(dataset, n_prev=look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-n_prev):
		dataX.append(dataset[i:(i+n_prev), :])
		dataY.append(dataset[(i+n_prev), :])
	return np.array(dataX), np.array(dataY)

def train_test_split(df, train_size=0.67):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (train_size))

    X_train, y_train = _load_data(df[0:ntrn])
    X_test, y_test = _load_data(df[ntrn:])
    #print(X_test.shape)
    #print("X_test=", X_test)
    #print(y_test.shape)
    #print("y_test=", y_test)
    return (X_train, y_train), (X_test, y_test)

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

(trainX, trainY), (testX, testY) = train_test_split(dataset)  # retrieve data

tbCallBack = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

# expected input data shape: (batch_size, timesteps, num_features)
model = Sequential()
model.add(LSTM(hidden_neurons, input_shape=(None,num_features), return_sequences=False))
model.add(Dense(num_features))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=num_epochs, batch_size=batch_size, verbose=2, validation_data=(testX, testY), callbacks=[tbCallBack])

#how well did it do? 
score = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
for j in range(len(score)):
    print("%s: %f" % (model.metrics_names[j], score[j]))

#Save the model
# serialize model to JSON
model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/model.h5")
print("Saved model to disk")

quit()

# make predictions
trainPredict = model.predict(trainX)
print(trainPredict.shape)
print(trainPredict)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict.reshape(-1, 1))
print("trainPredict.shape=", trainPredict.shape) 
print("trainY.shape=", trainY.shape) 
trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1))
print("testPredict.shape=", testPredict.shape) 
print("testY.shape=", testY.shape) 
testY = scaler.inverse_transform(testY.reshape(-1, 1))
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()



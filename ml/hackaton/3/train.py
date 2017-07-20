#!/usr/bin/env python

import json
import numpy as np
import math

# fix random seed for reproducibility.
# WARN: Open issue: Keras API reproducibility #11585
# https://github.com/tensorflow/tensorflow/issues/11585
#np.random.seed(7)
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.layers import LSTM
from tensorflow.contrib.keras.api.keras.callbacks import TensorBoard

# http://danielhnyk.cz/predicting-sequences-vectors-keras-using-rnn-lstm/
# https://github.com/trnkatomas/Keras_2_examples/blob/master/Simple_LSTM_keras_2.0.ipynb

num_features = 5
look_back = 3
moving_avg_size = 30
batch_size = 32
hidden_neurons=20
num_epochs=100
# dlen = 48 hours = 48*60 data points used in training
dlen=60*48
m = np.zeros((dlen, num_features), dtype=float)


def moving_avg(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def walk(d):
    t0=0
    for k in d['aggregations']['per_minute']['buckets']:
        timestamp=k['key']
        timeval=k['key_as_string']
        X = np.zeros(num_features, dtype=float)
        val=k['code_dist']['buckets']
        for i in range(num_features):
            X[i] = val[i]['doc_count']

        S = np.sum(X)
        if S > 0:
            X = X / S
        if t0 == 0:
            yield 0, X, timeval
            t0=timestamp
        else:
            yield (timestamp-t0)/1000, X, timeval


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
    print(X_test.shape)
    print("X_test=", X_test)
    print(y_test.shape)
    print("y_test=", y_test)
    return (X_train, y_train), (X_test, y_test)

with open('input.json') as json_data:
    j=0
    d = json.load(json_data)
    for key, val, timeval in walk(d):
        # print(timeval, val)
        m[j] = val
        j = j+1
        if j == dlen:
            break


dataset_ = np.zeros((dlen-moving_avg_size+1, num_features), dtype=float)
for j in range(num_features):
    dataset_[:,j]=moving_avg(m[:,j], moving_avg_size)

# No need to scale. Data is already in [0,1] range
dataset = dataset_

(trainX, trainY), (testX, testY) = train_test_split(dataset)  # retrieve data

tbCallBack = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

# expected input data shape: (batch_size, timesteps, num_features)
model = Sequential()
model.add(LSTM(hidden_neurons, input_shape=(None,num_features), return_sequences=False))
model.add(Dense(num_features, input_dim=hidden_neurons, activation='relu'))
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




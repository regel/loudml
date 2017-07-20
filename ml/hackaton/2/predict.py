#!/usr/bin/env python

#for matrix math
import numpy as np
#for importing our keras model
import tensorflow.contrib.keras.api.keras.models

import json

#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from load import *
#global vars for easy reusability
global model, graph

#initialize these variables
model, graph = init()

# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
num_features = 1
anomaly_threshold = 30.0
look_back = 5
max_mos = 5.0

max_dist=np.linalg.norm(np.zeros(num_features) - np.ones(num_features))

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

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

with open('input.json') as json_data:
    j = 0
    X = np.zeros((num_features,look_back), dtype=float)
    d = json.load(json_data)
    for _, val, timeval in walk(d):
        j = j+1
        X=np.roll(X,-1,axis=1)
        X[:,-1] = val / max_mos
        if j < look_back:
            continue

        try:
            # Y_ defined: compare the current value with model prediction
            mse = ((X[:,-1] - Y_.T[:,-1]) ** 2).mean(axis=None)
            dist = np.linalg.norm( (X[:,-1] - Y_.T[:,-1]) )
            #print("\ndist=", dist)
            #print("\nmse=", mse)
            score = (dist / max_dist) * 100
            if score > anomaly_threshold:
                print("Anomaly @timestamp:", timeval, "dist=", dist, "mse=", mse, "score=", score, "actual=", max_mos * X[:,-1].T, "predicted=", max_mos * Y_.T[:,-1].T)
        except NameError:
            # Y_ not defined means we don't have a model prediction yet
            mse=0

#        X_ = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        X_ = np.reshape(X.T, (1, look_back, num_features))
        Y_ = model.predict(X_, batch_size=1, verbose=0)



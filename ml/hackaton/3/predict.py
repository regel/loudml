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
num_features = 5
anomaly_threshold = 3.0
look_back = 3
moving_avg_size = 30

max_dist=np.linalg.norm(np.zeros(num_features) - np.ones(num_features))

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

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


with open('input.json') as json_data:
    j = 0
    X = np.zeros((num_features,look_back), dtype=float)
    Z = np.zeros((num_features, look_back+moving_avg_size-1), dtype=float)

    d = json.load(json_data)
    for _, val, timeval in walk(d):
        j = j+1
        Z=np.roll(Z,-1,axis=1)
        #print(val)
        Z[:,-1] = val 
        if j < (look_back+moving_avg_size-1):
            continue


        #print("Z=",Z)
        for l in range(num_features):
            X[l,:]=moving_avg(Z[l,:], moving_avg_size)
        
        # print("X=", X)

        try:
            # Y_ defined: compare the current value with model prediction
            mse = ((X[:,-1] - Y_.T[:,-1]) ** 2).mean(axis=None)
            dist = np.linalg.norm( (X[:,-1] - Y_.T[:,-1]) )
            #print("\ndist=", dist)
            #print("\nmse=", mse)
            score = (dist / max_dist) * 100
            if score > anomaly_threshold:
                print("Anomaly @timestamp:", timeval, "dist=", dist, "mse=", mse, "score=", score, "actual=", X[:,-1].T, "predicted=", Y_.T[:,-1].T)
        except NameError:
            # Y_ not defined means we don't have a model prediction yet
            mse=0

        X_ = np.reshape(X.T, (1, look_back, num_features))
        Y_ = model.predict(X_, batch_size=1, verbose=0)
        
        #print("X_=", X_)
        #print("Y_=", Y_.T)



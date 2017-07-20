#!/usr/bin/env python

import json
import numpy as np
import matplotlib.pyplot as plt
#import pandas
import math

# fix random seed for reproducibility
np.random.seed(7)
# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
look_back = 9

#X = np.array([[ 0.85081176,  0.86274983,  0.83904414,  0.81594797,  0.83741152]])
#print(X)
#print(X.shape)
#print(np.reshape(X, (X.shape[0], 1, X.shape[1])))
#quit()


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


a = np.random.rand(5,9)
b = np.zeros((5, 9-3+1), dtype=float)
print(a)
b[0,:]=moving_average(a[0,:])
b[1,:]=moving_average(a[1,:])
b[2,:]=moving_average(a[2,:])
b[3,:]=moving_average(a[3,:])
b[4,:]=moving_average(a[4,:])
print(b)


#plt.plot(a[0,:])
#plt.plot(b[0,:])
#plt.show()
#quit()

def walk(d):
    t0=0
    for k in d['aggregations']['per_minute']['buckets']:
        timestamp=k['key']
        timeval=k['key_as_string']
        X = np.zeros(5, dtype=float)
        val=k['code_dist']['buckets']
        X[0] = val[0]['doc_count']
        X[1] = val[1]['doc_count']
        X[2] = val[2]['doc_count']
        X[3] = val[3]['doc_count']
        X[4] = val[4]['doc_count']
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
    X = np.zeros((5,look_back), dtype=float)
    d = json.load(json_data)
    for _, val, timeval in walk(d):
        j = j+1
        X=np.roll(X,-1,axis=1)
        print(val)
        X[:,-1] = val 
        print(X)
        if j < look_back:
            continue
        X_ = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        Y_ = model.predict(X_, batch_size=1, verbose=0)
        print(X_, Y_)
        quit()


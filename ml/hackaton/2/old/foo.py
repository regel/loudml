#!/usr/bin/env python

import json
import numpy as np
import matplotlib.pyplot as plt
#import pandas
import math

# fix random seed for reproducibility
np.random.seed(7)
# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
look_back = 5
dlen=1000
m = np.zeros((dlen, 1), dtype=float)

#X = np.array([[ 0.85081176,  0.86274983,  0.83904414,  0.81594797,  0.83741152]])
#print(X)
#print(X.shape)
#print(np.reshape(X, (X.shape[0], 1, X.shape[1])))
#quit()



def walk(d):
    t0=0
    for k in d['aggregations']['per_minute']['buckets']:
        timestamp=k['key']
        val=k['avg_in_mos']['value']
        if val != None:
            if t0 == 0:
                yield 0, val
                t0=timestamp
            else:
                yield (timestamp-t0)/1000, val

with open('input.json') as json_data:
    X = np.zeros((1,look_back), dtype=float)
    
    d = json.load(json_data)
    for key, val in walk(d):
        X=np.roll(X,-1,axis=1)
        X[0][look_back-1] = val
        print(np.reshape(X, (X.shape[0], 1, X.shape[1])))
        #print(X)




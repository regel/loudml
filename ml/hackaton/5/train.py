#!/usr/bin/env python

import json
import numpy as np
import matplotlib.pyplot as plt
import math
# fix random seed for reproducibility.
np.random.seed(7)
#from tensorflow.contrib.keras.api.keras.models import Sequential
#from tensorflow.contrib.keras.api.keras.layers import Dense
#from tensorflow.contrib.keras.api.keras.layers import LSTM
#from tensorflow.contrib.keras.api.keras.callbacks import TensorBoard

def walk(d):
    t0=0
    for k in d['aggregations']['account']['buckets']:
        account=k['key']
        val=k['count']['buckets']
        if val != None:
            for l in val:
                timestamp=l['key']
                timeval=l['key_as_string']

with open('input.json') as json_data:
    j=0
    d = json.load(json_data)
    for key, val, timeval in walk(d):
        continue


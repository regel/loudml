#!/usr/bin/env python

import json
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import preprocessing

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})


days=30
timesteps=4*days
# features: [
#  in time range [0h-6h] : #call, avg duration, sdv duration. Repeat for international and premium calls.
#  in time range [6h-12h] : #call, avg duration, sdv duration. Repeat for international and premium calls.
#  in time range [12h-18h] : #call, avg duration, sdv duration. Repeat for international and premium calls.
#  in time range [18h-24h] : #call, avg duration, sdv duration. Repeat for international and premium calls.
#]

feature_names = [ \
     '0006_num_call','0006_avg_duration', '0006_sdv_duration',
     '0006_international_num_call','0006_international_avg_duration', '0006_international_sdv_duration',
     '0006_premium_num_call','0006_premium_avg_duration', '0006_premium_sdv_duration',
     '0612_num_call','0612_avg_duration', '0612_sdv_duration',
     '0612_international_num_call','0612_international_avg_duration', '0612_international_sdv_duration',
     '0612_premium_num_call','0612_premium_avg_duration', '0612_premium_sdv_duration',
     '1218_num_call','1218_avg_duration', '1218_sdv_duration',
     '1218_international_num_call','1218_international_avg_duration', '1218_international_sdv_duration',
     '1218_premium_num_call','1218_premium_avg_duration', '1218_premium_sdv_duration',
     '1824_num_call','1824_avg_duration', '1824_sdv_duration',
     '1824_international_num_call','1824_international_avg_duration', '1824_international_sdv_duration',
     '1824_premium_num_call','1824_premium_avg_duration', '1824_premium_sdv_duration'
    ]

num_features=len(feature_names)

# Calculate the long term (30 days) 'signatures' for each user account
def walk(d):
    t0=0
    for k in d['aggregations']['account']['buckets']:
        profile=np.zeros(num_features)
        account=k['key']
        val=k['count']['buckets']
        for l in val:
            timestamp=l['key']
            timeval=l['key_as_string']
            s=l['duration_stats']
            _count = float(s['count'])
            if _count == 0:
                continue
            quadrant = int( ((int(timestamp) / (3600*1000)) % 24)/6 )
            _min = float(s['min'])
            _max = float(s['max'])
            _avg = float(s['avg'])
            _sum = float(s['sum'])
            _sum_of_squares = float(s['sum_of_squares'])
            _variance = float(s['variance'])
            _std_deviation = float(s['std_deviation'])
            
            X = np.array( [_count, _sum, _sum_of_squares] )
            profile[(quadrant*9):(quadrant*9 +3)] += X

            if len(l['international']['buckets']) > 0:
                s=l['international']['buckets'][0]['duration_stats']
                _count = s['count']
                if _count == 0:
                    continue
                _min = float(s['min'])
                _max = float(s['max'])
                _avg = float(s['avg'])
                _sum = float(s['sum'])
                _sum_of_squares = float(s['sum_of_squares'])
                _variance = float(s['variance'])
                _std_deviation = float(s['std_deviation'])

                X = np.array( [_count, _sum, _sum_of_squares] )
                profile[(quadrant*9 +3):(quadrant*9 +6)] += X

            if len(l['premium']['buckets']) > 0:
                s=l['premium']['buckets'][0]['duration_stats']
                _count = s['count']
                if _count == 0:
                    continue
                _min = float(s['min'])
                _max = float(s['max'])
                _avg = float(s['avg'])
                _sum = float(s['sum'])
                _sum_of_squares = float(s['sum_of_squares'])
                _variance = float(s['variance'])
                _std_deviation = float(s['std_deviation'])

                X = np.array( [_count, _sum, _sum_of_squares] )
                profile[(quadrant*9 +3):(quadrant*9 +6)] += X

        for quadrant in range(4):
            for j in range(3):
                _count = profile[quadrant*9 + 3*j]
                _sum = profile[quadrant*9 + 3*j +1]
                _sum_of_squares = profile[quadrant*9 + 3*j +2]
                if _count > 0:
                    profile[quadrant*9 + 3*j +1] = _sum / _count
                    profile[quadrant*9 + 3*j +2] = math.sqrt(_sum_of_squares/_count - (_sum/_count)**2)
    
        #for j in range(len(feature_names)):
        #    print(feature_names[j], profile[j])
        yield account, profile


profiles = []
accounts = []
with open('input.json') as json_data:
    j=0
    d = json.load(json_data)
    for key, val in walk(d):
        print("key[%s]=" % key, val)
        profiles.append(val)
        accounts.append(key)
        j = j+1

profiles = np.array(profiles)

# Apply data standardization to each feature individually
# https://en.wikipedia.org/wiki/Feature_scaling 
# x_ = (x - mean(x)) / std(x)
# means = np.mean(profiles, axis=0)
# stds = np.std(profiles, axis=0)
profiles = preprocessing.scale(profiles)

# From: https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
import tensorflow as tf
import numpy as np
 # fix random seed for reproducibility.
np.random.seed(7)
from PIL import Image 

class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
 
    #To check if the SOM has been trained
    _trained = False
 
    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):
        """
        Initializes all necessary components of the TensorFlow
        Graph.
 
        m X n are the dimensions of the SOM. 'n_iterations' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        """
 
        #Assign required variables first
        self._dim = dim
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))
 
        ##INITIALIZE GRAPH
        self._graph = tf.Graph()
 
        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():
 
            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE
 
            #Randomly initialized weightage vectors for all neurons,
            #stored together as a matrix Variable of size [m*n, dim]
            self._weightage_vects = tf.Variable(tf.random_normal(
                [m*n, dim]))
 
            #Matrix of size [m*n, 2] for SOM grid locations
            #of neurons
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))
 
            ##PLACEHOLDERS FOR TRAINING INPUTS
            #We need to assign them as attributes to self, since they
            #will be fed in during training
 
            #The training vector
            self._vect_input = tf.placeholder("float", [dim])
            #Iteration number
            self._iter_input = tf.placeholder("float")
 
            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            #Only the final, 'root' training op needs to be assigned as
            #an attribute to self, since all the rest will be executed
            #automatically during training
 
            #To compute the Best Matching Unit given a vector
            #Basically calculates the Euclidean distance between every
            #neuron's weightage vector and the input, and returns the
            #index of the neuron which gives the least value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self._weightage_vects, tf.stack(
                    [self._vect_input for i in range(m*n)])), 2), 1)),
                                  0)
 
            #This will extract the location of the BMU based on the BMU's
            #index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]))),
                                 [2])
 
            #To compute the alpha and sigma values based on iteration
            #number
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input,
                                                  self._n_iterations))
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)
 
            #Construct the op that will generate a vector with learning
            #rates for all neurons, based on iteration number and location
            #wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vects, tf.stack(
                    [bmu_loc for i in range(m*n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)
 
            #Finally, the op that will use learning_rate_op to update
            #the weightage vectors of all neurons based on a particular
            #input
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                                               for i in range(m*n)])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vect_input for i in range(m*n)]),
                       self._weightage_vects))                                         
            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)                                       
 
            ##INITIALIZE SESSION
            self._sess = tf.Session()
 
            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            # 'Saver' op to save and restore all the variables
            self._saver = tf.train.Saver()
            self._sess.run(init_op)
 
    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        #Nested iterations over both dimensions
        #to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])
 
    def train(self, input_vects, verbose=1):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """
 
        #Training iterations
        for iter_no in range(self._n_iterations):
            if verbose>0 and (iter_no % 10 == 0):
                print("Training Iteration: ", iter_no)
            #Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})
 
        #Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid
 
        self._trained = True
 
    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid
 
    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """
 
        if not self._trained:
            raise ValueError("SOM not trained yet")
 
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect-
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])
 
        return to_return

    def save_model(self, model_path='/tmp'):
        if not self._trained:
            raise ValueError("SOM not trained yet")
        # Save model weights to disk
        save_path = self._saver.save(self._sess, model_path)
        print("Model saved in file: %s" % save_path)

    def restore_model(self, save_path=None):
        # Restore model weights to disk
        self._saver.restore(self._sess, save_path)
        print("Model restored from file: %s" % save_path)
        #Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid

        self._trained = True

    def show(self):
        """
        Displays the weight matrix as an RGB image
        """

        if not self._trained:
            raise ValueError("SOM not trained yet")

        X=256 * np.reshape(self._weightage_vects.eval(session=self._sess), (self._m, self._n, self._dim))
        im = Image.fromarray(X.astype('uint8'), mode='RGB')
        im.format = 'JPG'
        im.show()


# Hyperparameters
map_w = 50
map_h = 50
data_dimens = num_features
epochs = 100


# Defining Map
som = SOM(map_w, map_h, data_dimens, epochs)
# Start Training
som.train(profiles)
# FIXME: Hyperparameters are not saved, nor the model version
som.save_model('model/som_model.ckpt')


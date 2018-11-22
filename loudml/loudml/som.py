#!/usr/bin/env python

# TODO: clean-up and unit tests

# From: https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
import os
import logging
import multiprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from scipy.stats import norm

# fix random seed for reproducibility.
# np.random.seed(7)
from random import shuffle
from scipy import spatial
from functools import partial

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def get_nearest(tree, l):
    out=[]
    for x in l:
        distance, nearest = tree.query(x,k=1)
        out.append([distance, nearest])
    return out

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

flatten = lambda l: [item for sublist in l for item in sublist]

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


    def __enter__(self):
        return self

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

    def train(
        self,
        vects,
        verbose=1,
        truncate=-1,
        progress_cb=None,
    ):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """
        
        input_vects = list(vects)
        shuffle(input_vects)
        if truncate > 0:
            input_vects = input_vects[0:truncate]

        #Training iterations
        for iter_no in range(self._n_iterations):
            if progress_cb:
                progress_cb(iter_no, self._n_iterations)

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
 
        self._tree = spatial.cKDTree(self._weightages)
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
        batch = 256 # FIXME: find best value to distribute the workload
        if not self._trained:
            raise ValueError("SOM not trained yet")
 
        to_return = []
        if len(input_vects) > batch:
            pool = multiprocessing.Pool()
            func = partial(get_nearest, self._tree)
            for dd, ii in flatten(pool.map(func, list(chunks(input_vects, batch)))):
                to_return.append(self._locations[ii])
    
            pool.close()
            pool.join()
        else:
            for dd, ii in get_nearest(self._tree, input_vects):
                to_return.append(self._locations[ii])

        return to_return

    def get_scores(self,
                 y,
                 x,
                 low_highs,
        ):
        _norm = norm() 
        diff = x - y
        scores = 2 * _norm.cdf(abs(x - y)) - 1
        # Required to handle the 'low' condition
        scores[diff < 0] *= -1

        for i, ano_type in enumerate(low_highs):
            if ano_type == 'low':
                scores[i] = -min(scores[i], 0)
            elif ano_type == 'high':
                scores[i] = max(scores[i], 0)
            else:
                scores[i] = abs(scores[i])

        scores = 100 * np.clip(scores, 0, 1)
        return scores

    def location(self, x, y):
        return x * self._n + y

    def centroids(self):
        return self._weightages

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

        self._tree = spatial.cKDTree(self._weightages)
        self._trained = True

def serialize_model(som_model):
    import tempfile
    import base64
    from .som import SOM

    # serialize model to base64
    try:
        with tempfile.TemporaryDirectory() as tmp:
            som_model.save_model(os.path.join(tmp, 'model'))

            with open(os.path.join(tmp, 'model.data-00000-of-00001'), 'rb') as fp:
                data = base64.b64encode(fp.read())
            with open(os.path.join(tmp, 'model.index'), 'rb') as fp:
                index = base64.b64encode(fp.read())
            with open(os.path.join(tmp, 'model.meta'), 'rb') as fp:
                meta = base64.b64encode(fp.read())
    except OSError as exn:
        logging.error("cannot serialize SOM model: %s", str(exn))
        raise exn

    return data.decode('utf-8'), index.decode('utf-8'), meta.decode('utf-8')

def load_model(ckpt, index, meta, w, h, num_dimens):
    import tempfile
    import base64
    from .som import SOM

    try:
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, 'model.data-00000-of-00001'), 'wb') as fp:
                fp.write(base64.b64decode(ckpt.encode('utf-8')))
            with open(os.path.join(tmp, 'model.index'), 'wb') as fp:
                fp.write(base64.b64decode(index.encode('utf-8')))
            with open(os.path.join(tmp, 'model.meta'), 'wb') as fp:
                fp.write(base64.b64decode(meta.encode('utf-8')))

            # load weights into new model
            som_model = SOM(w, h, num_dimens, 0)
            som_model.restore_model(os.path.join(tmp, 'model'))
            return som_model
    except OSError as exn:
        logging.error("cannot load SOM model: %s", str(exn))
        raise exn

# Imports
import numpy as np
import math
from PIL import Image


class SOM:
    def __init__(self, x_size, y_size, dimen, num_iter):
        # init weights to 0 < w < 256
        self.weights = np.random.randint(256, size=(x_size, y_size, dimen))\
                            .astype('float64')
        self.num_iter = num_iter
        self.map_radius = max(self.weights.shape)/2 # sigma_0
        self.t_const = self.num_iter/math.log(self.map_radius) # lambda
        
    def get_bmu(self, vector):
        # calculate euclidean dist btw weight matrix and vector
        distance = np.sum((self.weights - vector) ** 2, 2)
        min_idx = distance.argmin()
        return np.unravel_index(min_idx, distance.shape)
        
    def get_bmu_dist(self, vector):
        # initialize array where values are its index
        x, y, rgb = self.weights.shape
        xi = np.arange(x).reshape(x, 1).repeat(y, 1)
        yi = np.arange(y).reshape(1, y).repeat(x, 0)
        # returns matrix of distance of each index in 2D from BMU
        return np.sum((np.dstack((xi, yi)) - np.array(self.get_bmu(vector))) ** 2, 2)

    def get_nbhood_radius(self, iter_count):
        return self.map_radius * np.exp(-iter_count/self.t_const)
        
    def teach_row(self, vector, i):
        nbhood_radius = self.get_nbhood_radius(i)
        bmu_dist = self.get_bmu_dist(vector).astype('float64')
        
        # exponential decaying learning rate
        lr = 0.1 * np.exp(-i/self.num_iter) 
        
        # influence
        theta = np.exp(-(bmu_dist)/ (2 * nbhood_radius ** 2))
        return np.expand_dims(theta, 2) * (vector - self.weights)
        
    def teach(self, t_set):
        for i in range(self.num_iter):
            if i % 10 == 0:
                print("Training Iteration: ", i)
            for j in range(len(t_set)):
                self.weights += self.teach_row(t_set[j], i)
        
    def show(self):
        im = Image.fromarray(self.weights.astype('uint8'), mode='RGB')
        im.format = 'JPG'
        im.show()


# Hyperparameters
map_w = 200
map_h = 200
data_dimens = 3
epochs = 100

# Initialize a random RGB training set 
training_set = np.random.randint(256, size=(15, 3))

# Defining Map
s = SOM(map_w, map_h, data_dimens, epochs)

# Start Training
s.teach(training_set)

# Display Result
s.show()




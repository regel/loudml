import loudml.vendor

import sys
import numpy as np

from scipy.stats import norm
from scipy.signal import argrelextrema
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KernelDensity

#import matplotlib.pyplot as plt

def get_groups(x):
    x = np.array(x)

    xd = np.linspace(np.min(x), np.max(x), 1000)
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        return_train_score=False,
                        cv=LeaveOneOut())
    grid.fit(x[:, None]);
    # print(grid.best_params_)
    
    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'], kernel='gaussian')
    kde.fit(x[:, None])
    # print(kde.get_params())
    
    # score_samples returns the log of the probability density
    logprob = kde.score_samples(xd[:, None])
    
    #plt.fill_between(xd, np.exp(logprob), alpha=0.5)
    #plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
    #plt.show()
    
    mi, ma = argrelextrema(logprob, np.less)[0], argrelextrema(logprob, np.greater)[0]
    # print( "Minima:", xd[mi])
    # print( "Maxima:", xd[ma])
    
    groups=[]
    groups.append((0, xd[mi][0]))
    for j in range(len(xd[mi]) - 1):
        groups.append((xd[mi][j], xd[mi][j+1]))
    
    groups.append((xd[mi][j+1], sys.float_info.max))
    return groups

#    plt.plot(xd[:mi[0]+1], logprob[:mi[0]+1], 'r',
#         xd[mi[0]:mi[1]+1], logprob[mi[0]:mi[1]+1], 'g',
#         xd[mi[1]:], logprob[mi[1]:], 'b',
#         xd[ma], logprob[ma], 'go',
#         xd[mi], logprob[mi], 'ro')
#    plt.show()
    

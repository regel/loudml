#
# keras and TF: When debugging, set seed to reproduce consistant output
from numpy.random import seed
from random import random
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)

import argparse
import logging
import json
import os
import sys
import sched, time

import numpy as np
import math

import tensorflow as tf
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.layers import Activation
from tensorflow.contrib.keras.api.keras.layers import LSTM
from tensorflow.contrib.keras.api.keras.callbacks import EarlyStopping

from hyperopt import hp
from hyperopt import space_eval
from hyperopt import fmin, tpe, STATUS_OK, Trials


class HyperParameters:
    def __init__(self):
        return
 
# global vars for easy reusability
# This UNIX process is handling a unique model
_model, _graph = None, None
_mins, _maxs = None, None
_verbose = 0

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

from .storage import (
    Storage,
)

get_current_time = lambda: int(round(time.time()))

import threading
from threading import current_thread

arg = None
threadLocal = threading.local()

def get_storage(elasticsearch_addr):
    global arg
    storage = getattr(threadLocal, 'storage', None)
    if storage is None:
        storage = Storage(elasticsearch_addr)
        threadLocal.storage = storage

    return storage

def log_message(format, *args):
    if len(request.remote_addr) > 0:
        addr = request.remote_addr
    else:
        addr = "-"

    sys.stdout.write("%s - - [%s] %s\n" % (addr,
                     # log_date_time_string()
                     "-", format % args))

def log_error(format, *args):
    log_message(format, *args)


def _load_data(dataset, n_prev=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-n_prev):
        if (np.isnan(dataset[i:(i+n_prev), :]).any() == False) and \
                (np.isnan(dataset[(i+n_prev), :]).any() == False):
            dataX.append(dataset[i:(i+n_prev), :])
            dataY.append(dataset[(i+n_prev), :])

    return np.array(dataX), np.array(dataY)

def train_test_split(df, n_prev=1, train_size=0.67):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (train_size))

    X_train, y_train = _load_data(df[0:ntrn], n_prev=n_prev)
    X_test, y_test = _load_data(df[ntrn:], n_prev=n_prev)
    return (X_train, y_train), (X_test, y_test)

def predict(
        model,
        from_date=None,
        to_date=None,
        anomaly_threshold=30,
    ): 
    global _model, _graph, _mins, _maxs

    num_features = len(model._features)
    n_prev = int(model._span / model._bucket_interval)

    # max_dist = sqrt(num_features) is the max Euclidian distance (norm2) in 
    # N dimension space when all values are in range [0,1]
    max_dist=np.linalg.norm(np.zeros(num_features) - np.ones(num_features))

    # Y_ (next predicted) must persist across API calls
    if model.Y_ is not None:
        Y_ = model.Y_

    _mse=[]
    _dist=[]
    _score=[]
    y_test=[]
    predicted=[]
    X = np.zeros((num_features, n_prev), dtype=float)
    j = 0
    for _, val, timeval in model.get_np_data(from_date=from_date, to_date=to_date):
        j = j+1
        y_test.append(val)
        rng = _maxs - _mins
        val = 1.0 - (((_maxs - val)) / rng)

        X = np.roll(X,-1,axis=1)
        X[:,-1] = val 
        if j < n_prev:
            # More data required to do a predict operation
            predicted.append([np.nan for x in range(num_features)])
            continue

        if (np.isnan(X).any() == True):
            # NaN. Don't predict when data is missing
            predicted.append([np.nan for x in range(num_features)])
            continue

        try:
            # Y_ defined: compare the current value with model prediction
            mse = ((X[:,-1] - Y_.T[:,-1]) ** 2).mean(axis=None)
            dist = np.linalg.norm( (X[:,-1] - Y_.T[:,-1]) )
            score = (dist / max_dist) * 100
            _mse.append(mse)
            _dist.append(dist)
            _score.append(score)
            if score > anomaly_threshold:
                # NOTE: A good spot for PagerDuty integration ?
                print("Anomaly @timestamp:", timeval,
                             "dist=", dist,
                             "mse=", mse,
                             "score=", score,
                             "actual=", X[:,-1].T,
                             "predicted=", Y_.T[:,-1].T)
        except NameError:
            # Y_ not defined means we don't have a model prediction yet
            _mse.append(0)
            _dist.append(0)
            _score.append(0)

        X_ = np.reshape(X.T, (1, n_prev, num_features))
        Y_ = _model.predict(X_, batch_size=1, verbose=_verbose)
        model.Y_ = Y_

        # min/max inverse operation
        Z_ = _maxs - rng * (1 - Y_)
        predicted.append(Z_[0])

    return (_mse, _dist, _score, np.array(y_test), np.array(predicted))

def mp_train_model(
        elasticsearch_addr,
        name,
        from_date=None,
        to_date=None,
    ):
    global _model, _graph, _mins, _maxs
    _model, _graph = None, None
    _mins, _maxs = None, None

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #initialize these variables
    storage = get_storage(elasticsearch_addr)
    model = storage.get_model(name)
    if model is None:
        logging.error('Cannot get model %s' % name)
        raise Exception('Missing model information')
    best_params, score, y_test, predicted = \
        train(model,
              from_date,
              to_date)

    model.save_model(_model, mins=_mins, maxs=_maxs, best_params=best_params)
    return score

def train(
        model,
        from_date=None,
        to_date=None,
        train_size=0.67,
        batch_size = 64,
        num_epochs=100,
        max_evals=10,
    ):
    global _model, _graph, _mins, _maxs
    _model, _graph = None, None
    _mins, _maxs = None, None

    logging.info('train(%s) range=[%s, %s] train_size=%f batch_size=%d epochs=%d)' \
                  % (model._name, str(time.ctime(from_date)), str(time.ctime(to_date)), train_size, batch_size, num_epochs))

    to_date = 1000 * int(to_date / model._bucket_interval) * model._bucket_interval
    from_date = 1000 * int(from_date / model._bucket_interval) * model._bucket_interval
    num_buckets = int( (to_date - from_date) / (1000 * model._bucket_interval) ) + 1
    num_features = len(model._features)
    n_prev = int(model._span / model._bucket_interval)
    dlen = num_buckets
    dataset = np.zeros((dlen, num_features), dtype=float)

    j = 0
    for _, val, timeval in model.get_np_data(from_date=from_date, to_date=to_date):
        dataset[j] = val
        j += 1

    logging.info('Found %d time period' % j)

    # Min-max preprocessing to bring data in interval (0,1)
    # FIXME: support other normalization techniques
    # Preprocess each column (axis=0)
    _mins = np.min(np.nan_to_num(dataset), axis=0)
    _maxs = np.max(np.nan_to_num(dataset), axis=0)
    rng = _maxs - _mins
    _dataset = 1.0 - (((_maxs - dataset)) / rng)

    logging.info('Preprocessing. mins: %s maxs: %s ranges: %s' % (_mins, _maxs, rng))

    (X_train, y_train), (X_test, y_test) = train_test_split(_dataset, n_prev=n_prev, train_size=train_size)

    def cross_val_model(params):
        global _model, _graph
        _model, _graph = None, None

        # Destroys the current TF graph and creates a new one.
        # Useful to avoid clutter from old models / layers.
        K.clear_session()

        # expected input data shape: (batch_size, timesteps, num_features)
        _model = Sequential()
        if (params.depth == 1):
            _model.add(LSTM(params.l1, input_shape=(None,num_features), return_sequences=False))
            _model.add(Dense(num_features, input_dim=params.l1))
        elif (params.depth == 2):
            _model.add(LSTM(params.l1, input_shape=(None,num_features), return_sequences=True))
            _model.add(LSTM(params.l2, return_sequences=False))
            _model.add(Dense(num_features, input_dim=params.l2))

        _model.add(Activation(params.activation))
        _model.compile(loss=params.loss_fct, optimizer=params.optimizer, metrics=['accuracy'])
    
        _stop = EarlyStopping(monitor='val_loss', patience=5, verbose=_verbose, mode='auto')
        _model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=_verbose, validation_data=(X_test, y_test), callbacks=[_stop])
        
        # How well did it do? 
        score = _model.evaluate(X_test, y_test, batch_size=batch_size, verbose=_verbose)
        for j in range(len(score)):
            print("%s: %f" % (_model.metrics_names[j], score[j]))
    
        return score
    
    hyperparameters = HyperParameters();

    # Parameter search space
    def objective(args):
        for key, value in args.items():
            try:
                if int(value) == value:
                    value = int(value)
                elif float(value) == value:
                    value = float(value)
            except(ValueError):
                pass
            setattr(hyperparameters, key, value)
        score = cross_val_model(hyperparameters)
        return {'loss': score[0], 'status': STATUS_OK}

    space = {
        'depth': hp.choice('depth', [1,2]),
        'l1': 1+hp.randint('l1', 100),
        'l2': 1+hp.randint('l2', 100),
        'activation': hp.choice('activation', ['tanh']),
        'loss_fct': hp.choice('loss_fct', ['mean_squared_error']),
        'optimizer': hp.choice('optimizer', ['adam']),
    }
    
    # The Trials object will store details of each iteration
    trials = Trials()

    # Run the hyperparameter search using the tpe algorithm
    best = fmin(objective,
            space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials)

    # Get the values of the optimal parameters
    best_params = space_eval(space, best)
    print('best_params=', best_params)
    hyperparameters = HyperParameters();
    for key, value in best_params.items():
        try:
            if int(value) == value:
                value = int(value)
            elif float(value) == value:
                value = float(value)
        except(ValueError):
            pass
        setattr(hyperparameters, key, value)

    score = cross_val_model(hyperparameters)

    predicted = _model.predict(X_test) 
    return (best_params, score, y_test[:], predicted[:])

# Debug function to plot input data and predictions
def plot(y_test, predicted, dimension):
    import matplotlib.pylab as plt
    plt.rcParams["figure.figsize"] = (17, 9)
    plt.plot(predicted[:,dimension],"--")
    plt.plot(y_test[:,dimension],":")
    plt.show()

def periodic(scheduler, interval, action, actionargs=()):
    scheduler.enter(interval, 1, periodic,
                    (scheduler, interval, action, actionargs))
    action(*actionargs)

def __predict(model, anomaly_threshold):
    tick = get_current_time()
    to_date = 1000 * int((tick - model._offset))
    from_date = (to_date - 1000 * model._span)
    predict(model,
            from_date=from_date,
            to_date=to_date,
            anomaly_threshold=anomaly_threshold)

def periodic_predict(
        model,
        from_date=None,
        to_date=None,
        anomaly_threshold=30,
        real_time=False,
    ):
    if from_date is not None and to_date is not None:
        to_date = 1000 * int(to_date / model._bucket_interval) * model._bucket_interval
        from_date = 1000 * int(from_date / model._bucket_interval) * model._bucket_interval
        _mse, _dist, _score, \
            y_test, predicted = predict(model,
                                        from_date,
                                        to_date,
                                        anomaly_threshold)

        for j in range(len(model._features)):
            plot(y_test, predicted, j)

    if (real_time == True):
        s = sched.scheduler(time.time, time.sleep)
        periodic(s, model._interval, __predict, (model, anomaly_threshold))
        s.run()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    global _model, _graph, _mins, _maxs
    global _verbose
    global arg
    parser = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'elasticsearch_addr',
        help="Elasticsearch address",
        type=str,
        nargs='?',
        default="localhost:9200",
    )
    parser.add_argument(
        '-m', '--model',
        help="Model name",
        type=str,
        default=None,
    )
    parser.add_argument(
        '-p', '--predict',
        help="Predict and raise anomalies",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
    )
    parser.add_argument(
        '--threshold',
        help="Anomaly threshold in range [0, 100]",
        type=int,
        default=30,
    )
    parser.add_argument(
        '-t', '--train',
        help="Train and save model",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
    )
    parser.add_argument(
        '-r', '--real_time',
        help="Predict using real time data",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
    )
    parser.add_argument(
        '--plot',
        help="Plot y_test vs predicted data",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
    )
    parser.add_argument(
        '-s', '--start',
        help="Start date",
        type=int,
        default=get_current_time(),
    )
    parser.add_argument(
        '-e', '--end',
        help="End date",
        type=int,
        default=None,
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Message verbosity level",
        type=int,
        default=0,
    )

    arg = parser.parse_args()
    _verbose = arg.verbose

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #initialize these variables
    storage = get_storage(arg.elasticsearch_addr)
    model = storage.get_model(arg.model)
    if model is None:
        logging.error('Cannot get model %s' % name)
        raise Exception('Missing model information')

    from_date = arg.start
    to_date = arg.end
    anomaly_threshold = arg.threshold

    if (arg.predict == True):
        _model, _graph, _mins, _maxs = model.load_model()
        periodic_predict(
            model,
            from_date=from_date,
            to_date=to_date,
            anomaly_threshold=anomaly_threshold,
            real_time=arg.real_time)
    elif (arg.train == True):
        if from_date is None:
            logging.error('Missing datetime argument')
            raise Exception('Missing argument')
        if to_date is None:
            to_date = get_current_time()
 
        best_params, score, y_test, predicted = train(model, from_date, to_date)
        for j in range(len(model._features)):
            if (arg.plot == True):
                plot(y_test, predicted, j)

        model.save_model(_model, mins=_mins, maxs=_maxs, best_params=best_params)

if __name__ == "__main__":
    # execute only if run as a script
    main()


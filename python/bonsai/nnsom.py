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
from sklearn import preprocessing

from .som import SOM
 
# global vars for easy reusability
# This UNIX process is handling a unique model
_model = None
_verbose = 0

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

from .storage import (
    Storage,
    _SUNSHINE_NUM_FEATURES,
    map_quadrant_names,
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

def nnsom_train_model(
        elasticsearch_addr,
        name,
        from_date=None,
        to_date=None,
        num_epochs=100,
    ):
    global _model
    _model = None

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #initialize these variables
    storage = get_storage(elasticsearch_addr)
    model = storage.get_nnsom(name)
    if model is None:
        logging.error('Cannot get model %s' % name)
        raise Exception('Missing model information')
    train(model,
          from_date,
          to_date,
          num_epochs=num_epochs,
          )

    model.save_model(_model)
    return

def train(
        model,
        from_date=None,
        to_date=None,
        num_epochs=100,
    ):
    global _model
    _model = None

    logging.info('train(%s) range=[%s, %s] epochs=%d)' \
                  % (model._name, str(time.ctime(from_date)), str(time.ctime(to_date)), num_epochs))

    to_date = 1000 * int(to_date / model._interval) * model._interval
    from_date = 1000 * int(from_date / model._interval) * model._interval

    profiles = []
    terms = []
    for key, val in model.get_profile_data(from_date=from_date, to_date=to_date):
        # print("key[%s]=" % key, val)
        profiles.append(val)
        terms.append(key)

    if (len(profiles) == 0):
        return None
    profiles = np.array(profiles)

    # Apply data standardization to each feature individually
    # https://en.wikipedia.org/wiki/Feature_scaling 
    # x_ = (x - mean(x)) / std(x)
    # means = np.mean(profiles, axis=0)
    # stds = np.std(profiles, axis=0)
    profiles = preprocessing.scale(profiles)

    logging.info('Found %d profiles' % len(profiles))
    # Hyperparameters
    data_dimens = _SUNSHINE_NUM_FEATURES
    _model = SOM(model._map_w, model._map_h, data_dimens, num_epochs)
    # Start Training
    _model.train(profiles)

    #Map profiles to their closest neurons
    mapped = _model.map_vects(profiles)

    X = {}
    for x in range(len(mapped)):
        term = terms[x]
        X[term] = [ mapped[x][0], mapped[x][1] ]

    # print(X)
    return X

def get_account(model,
            account_name,
            from_date=None,
            to_date=None,
    ):
    logging.info('get_account(%s) range=[%s, %s])' \
                  % (account_name, str(time.ctime(from_date)), str(time.ctime(to_date))))

    g=model.get_profile_data(from_date=from_date, to_date=to_date, account_name=account_name)
    try:
        key, val = next(g)
    except(StopIteration):
        return None

    Y = np.array(val)

    # Apply data standardization to each feature individually
    # https://en.wikipedia.org/wiki/Feature_scaling 
    # x_ = (x - mean(x)) / std(x)
    # means = np.mean(profiles, axis=0)
    # stds = np.std(profiles, axis=0)
    zY = preprocessing.scale(Y)
    res = { 'key': key,
             'time_range_ms': (from_date, to_date),
             'Y': map_quadrant_names(Y),
             'zY': map_quadrant_names(zY) }
    return res

def map_account(model,
            account_name,
            from_date=None,
            to_date=None,
    ):
    global _model

    logging.info('map_account(%s) range=[%s, %s])' \
                  % (account_name, str(time.ctime(from_date)), str(time.ctime(to_date))))

    g=model.get_profile_data(from_date=from_date, to_date=to_date, account_name=account_name)
    try:
        key, val = next(g)
    except(StopIteration):
        return None

    Y = np.array(val)

    # Apply data standardization to each feature individually
    # https://en.wikipedia.org/wiki/Feature_scaling 
    # x_ = (x - mean(x)) / std(x)
    # means = np.mean(profiles, axis=0)
    # stds = np.std(profiles, axis=0)
    zY = preprocessing.scale(Y)
    #Map profile to its closest neurons
    mapped = _model.map_vects(zY)

    res = { 'key': key,
             'time_range_ms': (from_date, to_date),
             'Y': map_quadrant_names(Y),
             'zY': map_quadrant_names(zY),
             'mapped': ( mapped[0][0].item(), mapped[0][1].item() ),
             'dimension': ( model._map_w, model._map_h ),
           }
    return res

def map_accounts(model,
            from_date=None,
            to_date=None,
    ):
    global _model

    logging.info('map_accounts() range=[%s, %s])' \
                  % (str(time.ctime(from_date)), str(time.ctime(to_date))))

    res = []
    for key, val in model.get_profile_data(from_date=from_date, to_date=to_date):
        # print("key[%s]=" % key, val)

        Y = np.array(val)
    
        # Apply data standardization to each feature individually
        # https://en.wikipedia.org/wiki/Feature_scaling 
        # x_ = (x - mean(x)) / std(x)
        # means = np.mean(profiles, axis=0)
        # stds = np.std(profiles, axis=0)
        zY = preprocessing.scale(Y)
        #Map profile to its closest neurons
        mapped = _model.map_vects(zY)
    
        res.append({ 'key': key,
                 'time_range_ms': (from_date, to_date),
                 'Y': map_quadrant_names(Y),
                 'zY': map_quadrant_names(zY),
                 'mapped': ( mapped[0][0].item(), mapped[0][1].item() ),
                 'dimension': ( model._map_w, model._map_h ),
               })
    return res


def async_map_account(
        elasticsearch_addr,
        name,
        account_name,
        from_date=None,
        to_date=None,
    ):
    global _model
    _model = None

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #initialize these variables
    storage = get_storage(elasticsearch_addr)
    model = storage.get_nnsom(name)
    if model is None:
        logging.error('Cannot get model %s' % name)
        raise Exception('Missing model information')

    if (model.is_trained() == False):
        logging.error('Not yet trained: %s' % name)
        raise Exception('Missing training data')

    _model = model.load_model()

    to_date = 1000 * int(to_date / model._interval) * model._interval
    from_date = 1000 * int(from_date / model._interval) * model._interval

    return map_account(model,
          account_name=account_name,
          from_date=from_date,
          to_date=to_date,
          )

def async_map_accounts(
        elasticsearch_addr,
        name,
        from_date=None,
        to_date=None,
    ):
    global _model
    _model = None

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #initialize these variables
    storage = get_storage(elasticsearch_addr)
    model = storage.get_nnsom(name)
    if model is None:
        logging.error('Cannot get model %s' % name)
        raise Exception('Missing model information')

    if (model.is_trained() == False):
        logging.error('Not yet trained: %s' % name)
        raise Exception('Missing training data')

    _model = model.load_model()

    to_date = 1000 * int(to_date / model._interval) * model._interval
    from_date = 1000 * int(from_date / model._interval) * model._interval

    return map_accounts(model,
          from_date=from_date,
          to_date=to_date,
          )


def predict(model,
            from_date=None,
            to_date=None,
            anomaly_threshold=30,
    ):
    global _model

    logging.info('predict(%s) range=[%s, %s] threshold=%d)' \
                  % (model._name, str(time.ctime(from_date)), str(time.ctime(to_date)), anomaly_threshold))

    profiles = []
    terms = []
    for key, val in model.get_profile_data(from_date=from_date, to_date=to_date):
        # print("key[%s]=" % key, val)
        profiles.append(val)
        terms.append(key)
    
    if (len(profiles) == 0):
        return

    profiles = np.array(profiles)

    # Apply data standardization to each feature individually
    # https://en.wikipedia.org/wiki/Feature_scaling 
    # x_ = (x - mean(x)) / std(x)
    # means = np.mean(profiles, axis=0)
    # stds = np.std(profiles, axis=0)
    profiles = preprocessing.scale(profiles)

    #Map profiles to their closest neurons
    mapped = _model.map_vects(profiles)
    X = {}
    for x in range(len(mapped)):
        term = terms[x]
        X[term] = [ mapped[x][0], mapped[x][1] ]

    # FIXME: measure distance and score

    return  

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

def nnsom_rt_predict(
        elasticsearch_addr,
        name,
        anomaly_threshold=30,
    ):
    global _model
    _model = None

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #initialize these variables
    storage = get_storage(elasticsearch_addr)
    model = storage.get_nnsom(name)
    if model is None:
        logging.error('Cannot get model %s' % name)
        raise Exception('Missing model information')

    if (model.is_trained() == False):
        logging.error('Not yet trained: %s' % name)
        raise Exception('Missing training data')

    _model = model.load_model()

    s = sched.scheduler(time.time, time.sleep)
    periodic(s, model._interval, __predict, (model, anomaly_threshold))
    s.run()


def periodic_predict(
        model,
        from_date=None,
        to_date=None,
        anomaly_threshold=30,
        real_time=False,
    ):
    if from_date is not None and to_date is not None:
        to_date = 1000 * int(to_date / model._interval) * model._interval
        from_date = 1000 * int(from_date / model._interval) * model._interval
        predict(model,
                from_date,
                to_date,
                anomaly_threshold)

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
    global _model
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
    parser.add_argument(
        '--map_w',
        help="SOM width",
        type=int,
        default=50,
    )
    parser.add_argument(
        '--map_h',
        help="SOM height",
        type=int,
        default=50,
    )
    parser.add_argument(
        '--num_epochs',
        help="Epochs used in training",
        type=int,
        default=100,
    )

    arg = parser.parse_args()
    _verbose = arg.verbose

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #initialize these variables
    storage = get_storage(arg.elasticsearch_addr)
    model = storage.get_nnsom(arg.model)
    if model is None:
        logging.error('Cannot get model %s' % name)
        raise Exception('Missing model information')

    from_date = arg.start
    to_date = arg.end
    anomaly_threshold = arg.threshold

    if (arg.predict == True):
        _model = model.load_model()
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
 
        train(model,
              from_date,
              to_date,
              num_epochs=arg.num_epochs,
              )

        model.save_model(_model)

if __name__ == "__main__":
    # execute only if run as a script
    main()


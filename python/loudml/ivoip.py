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
import base64

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

def distance(x,
             y,
    ):
    dim = np.array(x['dimension'])
    x = np.array(x['mapped'])
    y = np.array(y['mapped'])
    # norm2 
    max_norm = np.linalg.norm(dim)
    dist = np.linalg.norm(x-y)
    score = int(100 * dist / max_norm) if max_norm > 0 else 0
    res = {
              'distance': dist,
              'score': score,
          }
    return res

def ivoip_train_model(
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
    model = storage.get_ivoip(name)
    if model is None:
        logging.error('Cannot get model %s' % name)
        raise Exception('Missing model information')
    mapped_info = train(model,
          from_date,
          to_date,
          num_epochs=num_epochs,
          )

    model.save_model(_model, mapped_info)
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

    Y = []
    terms = []
    for key, val in model.get_profile_data(from_date=from_date, to_date=to_date):
        # print("key[%s]=" % key, val)
        Y.append(val)
        terms.append(key)

    if (len(Y) == 0):
        return None
    Y = np.array(Y)

    # Apply data standardization to each feature individually
    # https://en.wikipedia.org/wiki/Feature_scaling 
    # x_ = (x - mean(x)) / std(x)
    # means = np.mean(profiles, axis=0)
    # stds = np.std(profiles, axis=0)
    zY = preprocessing.scale(Y)

    logging.info('Found %d profiles' % len(Y))
    # Hyperparameters
    data_dimens = _SUNSHINE_NUM_FEATURES
    _model = SOM(model._map_w, model._map_h, data_dimens, num_epochs)
    # Start Training
    _model.train(zY)

    #Map profiles to their closest neurons
    mapped = _model.map_vects(zY)
    mapped_info = []
    for x in range(len(mapped)):
        key = terms[x]
        mapped_info.append({ 'key': key,
             'time_range_ms': (from_date, to_date),
             'Y': Y[x].tolist(),
             'zY': zY[x].tolist(),
             'mapped': ( mapped[x][0].item(), mapped[x][1].item() ),
           })

    return mapped_info

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
             'Y': Y.tolist(),
             'zY': zY.tolist() }
    return res

def map_account(model,
            account_name,
            from_date=None,
            to_date=None,
    ):
    global _model

    logging.info('map_account(%s) range=[%s, %s])' \
                  % (account_name, str(time.ctime(from_date/1000)), str(time.ctime(to_date/1000))))

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
             'Y': Y.tolist(),
             'zY': zY.tolist(),
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
                  % (str(time.ctime(from_date/1000)), str(time.ctime(to_date/1000))))

    stored = stored_accounts(model)
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

        mapped_res = { 'key': key,
                 'time_range_ms': (from_date, to_date),
                 'Y': Y.tolist(),
                 'zY': zY.tolist(),
                 'mapped': ( mapped[0][0].item(), mapped[0][1].item() ),
                 'dimension': ( model._map_w, model._map_h ),
               }
        if key in stored:
            diff = distance(mapped_res, stored[key])
        else:
            diff = None
        res.append({'current': mapped_res, 'orig': stored[key], 'diff': diff})

    return res

def stored_account(model,
            key,
    ):
    if not 'mapped_info' in model._state:
        return None

    enc = model._state['mapped_info']
    object_list = json.loads(base64.b64decode(enc.encode('utf-8')).decode('utf-8'))
    mapped_info = dict((x['key'], x) for x in object_list)
    if key in mapped_info:
        return mapped_info[key]
    else:
        return None

def stored_accounts(model,
    ):
    if not 'mapped_info' in model._state:
        return None

    enc = model._state['mapped_info']
    object_list = json.loads(base64.b64decode(enc.encode('utf-8')).decode('utf-8'))
    mapped_info = dict((x['key'], x) for x in object_list)
    return mapped_info

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
    logging.info('async_map_account() range=[%s, %s])' \
                  % (str(time.ctime(from_date)), str(time.ctime(to_date))))

    #initialize these variables
    storage = get_storage(elasticsearch_addr)
    model = storage.get_ivoip(name)
    if model is None:
        logging.error('Cannot get model %s' % name)
        raise Exception('Missing model information')

    if (model.is_trained() == False):
        logging.error('Not yet trained: %s' % name)
        raise Exception('Missing training data')

    _model = model.load_model()

    to_date = 1000 * int(to_date / model._interval) * model._interval
    from_date = 1000 * int(from_date / model._interval) * model._interval

    mapped = map_account(model,
                         account_name=account_name,
                         from_date=from_date,
                         to_date=to_date,
                         )
    stored = stored_account(model, account_name)
    diff = distance(mapped, stored)
    return { 'current': mapped, 'orig': stored, 'diff': diff }

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
    logging.info('async_map_accounts() range=[%s, %s])' \
                  % (str(time.ctime(from_date)), str(time.ctime(to_date))))

    #initialize these variables
    storage = get_storage(elasticsearch_addr)
    model = storage.get_ivoip(name)
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

def async_score_hist(
        elasticsearch_addr,
        name,
        from_date=None,
        to_date=None,
        span=None,
        interval=None,
    ):
    global _model
    _model = None

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #initialize these variables
    storage = get_storage(elasticsearch_addr)
    model = storage.get_ivoip(name)
    if model is None:
        logging.error('Cannot get model %s' % name)
        raise Exception('Missing model information')

    if (model.is_trained() == False):
        logging.error('Not yet trained: %s' % name)
        raise Exception('Missing training data')

    _model = model.load_model()

    bins = np.linspace(0, 100, 11)

    _start = int(from_date / model._interval) * model._interval
    _end = int(to_date / model._interval) * model._interval
    res = []
    while _start < _end:
        _from_date = (_start - span)
        _to_date = _start
        val = map_accounts(model,
              from_date=1000*_from_date,
              to_date=1000*_to_date,
              )
        data = []
        for i in val:
            data.append(i['diff']['score'])
        h = np.histogram(data, bins, weights=data)[0]
        res.append({'timestamp': _from_date, 'counts': h.tolist()})
        _start = _start + interval

    return {'bins': bins.tolist(), 'histogram': res }

def predict(model,
            from_date=None,
            to_date=None,
    ):
    global _model

    logging.info('predict(%s) range=[%s, %s] threshold=%d)' \
                  % (model._name, str(time.ctime(from_date/1000)), str(time.ctime(to_date/1000)), model._threshold))

    val = map_accounts(model,
          from_date=from_date,
          to_date=to_date,
          )
    for k in val:
        key = k['orig']['key']
        score = k['diff']['score']
        if score > model._threshold:
            # NOTE: A good spot for PagerDuty integration ?
            print("Anomaly @timestamp:", get_current_time(),
                         "score=", score,
                         "original=", k['orig']['mapped'],
                         "current=", k['current']['mapped'],
                         )

    return  

def periodic(scheduler, interval, action, actionargs=()):
    scheduler.enter(interval, 1, periodic,
                    (scheduler, interval, action, actionargs))
    action(*actionargs)

def __predict(model):
    tick = get_current_time()
    to_date = 1000 * int((tick - model._offset))
    from_date = (to_date - 1000 * model._span)
    predict(model,
            from_date=from_date,
            to_date=to_date,
            )

def ivoip_rt_predict(
        elasticsearch_addr,
        name,
    ):
    global _model
    _model = None

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #initialize these variables
    storage = get_storage(elasticsearch_addr)
    model = storage.get_ivoip(name)
    if model is None:
        logging.error('Cannot get model %s' % name)
        raise Exception('Missing model information')

    if (model.is_trained() == False):
        logging.error('Not yet trained: %s' % name)
        raise Exception('Missing training data')

    _model = model.load_model()

    s = sched.scheduler(time.time, time.sleep)
    periodic(s, model._interval, __predict, (model,))
    s.run()


def periodic_predict(
        model,
        from_date=None,
        to_date=None,
        real_time=False,
    ):
    if from_date is not None and to_date is not None:
        to_date = 1000 * int(to_date / model._interval) * model._interval
        from_date = 1000 * int(from_date / model._interval) * model._interval
        predict(model,
                from_date,
                to_date,
                )

    if (real_time == True):
        s = sched.scheduler(time.time, time.sleep)
        periodic(s, model._interval, __predict, (model,))
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
        default=70,
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
    model = storage.get_ivoip(arg.model)
    if model is None:
        logging.error('Cannot get model %s' % name)
        raise Exception('Missing model information')

    from_date = arg.start
    to_date = arg.end

    if (arg.predict == True):
        _model = model.load_model()
        periodic_predict(
            model,
            from_date=from_date,
            to_date=to_date,
            real_time=arg.real_time)
    elif (arg.train == True):
        if from_date is None:
            logging.error('Missing datetime argument')
            raise Exception('Missing argument')
        if to_date is None:
            to_date = get_current_time()
 
        mapped_info = train(model,
              from_date,
              to_date,
              num_epochs=arg.num_epochs,
              )

        model.save_model(_model, mapped_info)

if __name__ == "__main__":
    # execute only if run as a script
    main()


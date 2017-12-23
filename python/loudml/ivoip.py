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
import time

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import math

from .som import SOM

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def timing_val(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print( '%s took %0.3f ms' % (func.__name__, (t2 - t1)*1000.0) )
        return res, (t2 - t1)
    return wrapper
 
# global vars for easy reusability
# This UNIX process is handling a unique model
_model = None
_means = None
_stds = None
_verbose = 0

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

time_span = lambda x: x['time_range_ms'][1] - x['time_range_ms'][0] 
time_from = lambda x: x['time_range_ms'][0]
time_to = lambda x: x['time_range_ms'][1]
time_exist = lambda x: 'time_range_ms' in x

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


def annotate_col(j):
    quadrants = ["00h-06h", "06h-12h", "12h-18h", "18h-24h"]
    features = ["count(total)", "avg(duration)", "std(duration)",
                "count(international)", "avg(international.duration)", "std(international.duration)",
                "count(premium)", "avg(premium.duration)", "std(premium.duration)",
               ]
    quadrant = quadrants[int(j / len(features))]
    feature = features[int(j % len(features))]
    return quadrant, feature

def get_description(Yc, Yo, yc, yo):
    Yc = np.array(Yc)
    Yo = np.array(Yo)
    yc = np.array(yc)
    yo = np.array(yo)
    try:
        diff = (Yc - Yo) / np.abs(Yo)
        diff[np.isinf(diff)] = np.nan
        max_arg = np.nanargmax(np.abs(diff))
        max_val = diff[max_arg]
        quadrant, feature = annotate_col(max_arg)
        mul = int(max_val)
        if mul < 0:
            kind, desc = 'Low', 'Low %s in range %s. %d times lower than usual. Actual=%f Typical=%f' \
                             % (feature, quadrant, abs(mul), yc[max_arg], yo[max_arg])
        else:
            kind, desc = 'High', 'High %s in range %s. %d times higher than usual. Actual=%f Typical=%f' \
                             % (feature, quadrant, abs(mul), yc[max_arg], yo[max_arg]) 
    except(ValueError):
        kind, desc = 'None', 'None'
    return kind, desc

def async_ivoip_train_model(
        elasticsearch_addr,
        name,
        from_date=None,
        to_date=None,
        num_epochs=100,
        limit=-1,
    ):
    global _model
    global _means
    global _stds
    _model = None
    _means = None
    _stds = None

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #initialize these variables
    storage = get_storage(elasticsearch_addr)
    model = storage.get_ivoip(name)
    if model is None:
        logging.error('Cannot get model %s' % name)
        raise Exception('Missing model information')
    mapped_info, took = train(model,
          from_date,
          to_date,
          num_epochs=num_epochs,
          limit=limit,
          )

    model.save_model(_model, _means, _stds, mapped_info)
    return { 'took': int(took*1000) }

@timing_val
def train(
        model,
        from_date=None,
        to_date=None,
        num_epochs=100,
        limit=-1,
    ):
    global _model
    global _means
    global _stds
    _model = None
    _means = None
    _stds = None

    logging.info('train(%s) range=[%s, %s] epochs=%d limit=%d)' \
                  % (model._name, str(time.ctime(from_date)), str(time.ctime(to_date)), num_epochs, limit))

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
    _means = np.mean(Y, axis=0)
    _stds = np.std(Y, axis=0)
    zY = np.nan_to_num((Y - _means) / _stds)

    logging.info('Found %d profiles' % len(Y))
    # Hyperparameters
    data_dimens = _SUNSHINE_NUM_FEATURES
    _model = SOM(model._map_w, model._map_h, data_dimens, num_epochs)
    # Start Training
    _model.train(zY, truncate=limit)

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

@timing_val
def map_account(model,
            account_name,
            from_date=None,
            to_date=None,
    ):
    global _model
    global _means
    global _stds

    logging.info('map_account(%s) range=[%s, %s])' \
                  % (account_name, str(time.ctime(from_date/1000)), str(time.ctime(to_date/1000))))

    g=model.get_profile_data(from_date=from_date, to_date=to_date, account_name=account_name)
    try:
        key, val = next(g)
    except(StopIteration):
        return None

    Y = [val]
    Y = np.array(Y)

    # Apply data standardization to each feature individually
    # https://en.wikipedia.org/wiki/Feature_scaling 
    zY = np.nan_to_num((Y - _means) / _stds)

    #Map profile to its closest neurons
    mapped = _model.map_vects(zY)

    res = { 'key': key,
             'time_range_ms': (from_date, to_date),
             'Y': Y[0].tolist(),
             'zY': zY[0].tolist(),
             'mapped': ( mapped[0][0].item(), mapped[0][1].item() ),
             'dimension': ( model._map_w, model._map_h ),
           }
    return res

@timing_val
def map_accounts(model,
            from_date=None,
            to_date=None,
    ):
    global _model
    global _means
    global _stds
    batch = 1024

    logging.info('map_accounts() range=[%s, %s])' \
                  % (str(time.ctime(from_date/1000)), str(time.ctime(to_date/1000))))

    stored = stored_accounts(model)
    res = []
    for l in chunks(list(model.get_profile_data(from_date=from_date, to_date=to_date)), batch):
        lY = [val[1] for val in l]
        lY = np.array(lY)
        lzY = np.nan_to_num((lY - _means) / _stds)

        # Map profile to their closest neurons
        _mapped = _model.map_vects(lzY)

        for key_val, Y, zY, mapped in zip(l, lY, lzY, _mapped):
            key, val = key_val
            mapped_res = { 'key': key,
                     'time_range_ms': (from_date, to_date),
                     'Y': Y.tolist(),
                     'zY': zY.tolist(),
                     'mapped': ( mapped[0].item(), mapped[1].item() ),
                     'dimension': ( model._map_w, model._map_h ),
                   }

            try:
                orig = stored[key]
                diff = _model.distance(mapped_res['mapped'], orig['mapped'])
            except KeyError:
                orig = {}
                diff = None
            res.append({'current': mapped_res, 'orig': orig, 'diff': diff})

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

def async_ivoip_map_account(
        elasticsearch_addr,
        name,
        account_name,
        from_date=None,
        to_date=None,
    ):
    global _model
    global _means
    global _stds
    _model = None
    _means = None
    _stds = None

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info('async_ivoip_map_account() range=[%s, %s])' \
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

    _model, _means, _stds = model.load_model()

    to_date = 1000 * int(to_date / model._interval) * model._interval
    from_date = 1000 * int(from_date / model._interval) * model._interval

    mapped, took = map_account(model,
                         account_name=account_name,
                         from_date=from_date,
                         to_date=to_date,
                         )
    stored = stored_account(model, account_name)
    diff = _model.distance(mapped['mapped'], stored['mapped'])
    return { 'took': int(took*1000), 'current': mapped, 'orig': stored, 'diff': diff }

def async_ivoip_map_accounts(
        elasticsearch_addr,
        name,
        from_date=None,
        to_date=None,
    ):
    global _model
    global _means
    global _stds
    _model = None
    _means = None
    _stds = None

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info('async_ivoip_map_accounts() range=[%s, %s])' \
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

    _model, _means, _stds  = model.load_model()

    to_date = 1000 * int(to_date / model._interval) * model._interval
    from_date = 1000 * int(from_date / model._interval) * model._interval

    res, took = map_accounts(model,
          from_date=from_date,
          to_date=to_date,
          )
    return res

def async_ivoip_score_hist(
        elasticsearch_addr,
        name,
        from_date=None,
        to_date=None,
        span=None,
        interval=None,
    ):
    global _model
    global _means
    global _stds
    _model = None
    _means = None
    _stds = None

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

    _model, _means, _stds  = model.load_model()

    bins = np.linspace(0, 100, 11)

    _start = int(from_date / model._interval) * model._interval
    _end = int(to_date / model._interval) * model._interval
    res = []
    while _start < _end:
        _from_date = (_start - span)
        _to_date = _start
        val, took = map_accounts(model,
              from_date=1000*_from_date,
              to_date=1000*_to_date,
              )
        data = []
        for i in val:
            if i['diff'] is not None:
                data.append(i['diff']['score'])
        h = np.histogram(data, bins, weights=data)[0]
        res.append({'timestamp': _from_date, 'counts': h.tolist()})
        _start = _start + interval

    return {'bins': bins.tolist(), 'histogram': res }

@timing_val
def predict(model,
            from_date=None,
            to_date=None,
    ):
    global _model
    global _means
    global _stds

    logging.info('predict(%s) range=[%s, %s] threshold=%d)' \
                  % (model._name, str(time.ctime(from_date/1000)), str(time.ctime(to_date/1000)), model._threshold))

    val, took = map_accounts(model,
          from_date=from_date,
          to_date=to_date,
          )

    anomalies = {}
    changed = []
    for k in val:
        if not ('key' in k['orig']):
            # signature = initial
            k['current']['time_range_ms'] = (to_date, to_date)
            changed.append(k['current'])
            continue

        if time_span(k['orig']) < (1000 * model._span):
            # signature = extended range (_span)
            k['current']['time_range_ms'] = (time_from(k['orig']), to_date)
            changed.append(k['current'])

        key = k['orig']['key']
        score = k['diff']['score']
        if score > model._threshold:
            kind, desc = get_description(k['current']['zY'],
                                         k['orig']['zY'],
                                         k['current']['Y'],
                                         k['orig']['Y'])
            anomalies[key] = { '@timestamp': to_date,
                         'until': to_date,
                         'key': key,
                         'score': score,
                         'max_score': score,
                         'model': model._name,
                         'uuid': '',
                         'kind': kind, 
                         'description': desc, 
                       }

    model.set_anomalies(anomalies)

    # signature = full range (train_span)
    train_span = max([time_span(x['orig']) for x in val if time_exist(x['orig'])])
    not_trained = [x['orig']['key'] for x in val if time_exist(x['orig']) \
                              and time_span(x['orig']) < train_span \
                              and time_from(x['orig']) < (to_date - train_span)]

    for key in not_trained:
        mapped, took = map_account(model,
                         account_name=key,
                         from_date=(to_date - train_span),
                         to_date=to_date,
                         )
        changed.append(mapped)

    # Write changes to configuration
    if len(changed) > 0:
        stored = stored_accounts(model)
        d = { k['key']:k for k in changed }
        stored.update(d)
        model.set_mapped_info(list(stored.values()))

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

def async_ivoip_live_predict(
        elasticsearch_addr,
        name,
    ):
    global _model
    global _means
    global _stds
    _model = None
    _means = None
    _stds = None

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

    _model, _means, _stds  = model.load_model()

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

def replay_predict(
        model,
        from_date,
        to_date,
    ):
    _start = int(from_date / model._interval) * model._interval
    _end = int(to_date / model._interval) * model._interval
    while (_start + model._span) < _end:
        _from = 1000 * _start
        _to = 1000 * (_start + model._span)
        predict(model,
                from_date=_from,
                to_date=_to,
                )
        _start = _start + model._interval

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    global _model
    global _means
    global _stds
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
        '-R', '--replay',
        help="Replay history data",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
    )
    parser.add_argument(
        '-i', '--interval',
        help="Inference periodic interval",
        type=int,
        default=0,
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
    parser.add_argument(
        '-l', '--limit',
        help="Limit profiles count used in training",
        type=int,
        default=-1,
    )

    arg = parser.parse_args()
    _verbose = arg.verbose

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #initialize these variables
    storage = get_storage(arg.elasticsearch_addr)
    storage.set_threshold(
        name=arg.model,
        threshold=int(arg.threshold))
    if arg.interval > 0:
        storage.set_interval(
            name=arg.model,
            interval=int(arg.interval))

    model = storage.get_ivoip(arg.model)
    if model is None:
        logging.error('Cannot get model %s' % name)
        raise Exception('Missing model information')

    from_date = arg.start
    to_date = arg.end

    if (arg.predict == True):
        _model, _means, _stds  = model.load_model()
        if (arg.replay == True):
            replay_predict(
                model,
                from_date=from_date,
                to_date=to_date)
        else:
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
 
        mapped_info, took = train(model,
              from_date,
              to_date,
              num_epochs=arg.num_epochs,
              limit=arg.limit,
              )

        model.save_model(_model, _means, _stds, mapped_info)

if __name__ == "__main__":
    # execute only if run as a script
    main()


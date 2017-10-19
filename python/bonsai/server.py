import argparse
import datetime
import time
import base64
import logging
import json
import signal
import os
import sys

import multiprocessing
from multiprocessing import Pool
from multiprocessing import TimeoutError 

from .compute import mp_train_model
from .compute import range_predict
from .compute import rt_predict

from .nnsom import nnsom_train_model
from .nnsom import async_map_account
from .nnsom import nnsom_rt_predict

get_current_time = lambda: int(round(time.time()))

from flask import (
    Flask,
    g,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
)

from .storage import (
    HTTPError,
    StorageException,
    Storage,
)


import threading
from threading import current_thread

g_elasticsearch_addr = None
g_job_id = 0
g_processes = {}
g_jobs = {}
g_pool = None
arg = None
threadLocal = threading.local()

app = Flask(__name__, static_url_path='/static', template_folder='templates')

def get_storage():
    global g_elasticsearch_addr
    storage = getattr(threadLocal, 'storage', None)
    if storage is None:
        storage = Storage(g_elasticsearch_addr)
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

@app.errorhandler(HTTPError)
def exn_handler(exn):
    response = jsonify({
        'error': "Internal",
    })
    response.status_code = 500
    return response

def error_msg(msg, code):
    response = jsonify({
        'error': msg,
    })
    response.status_code = code
    return response


def terminate_job(job_id, timeout):
    global g_jobs
    try:
        res = g_jobs[job_id].wait(timeout)
    except(TimeoutError):
        g_jobs[job_id].terminate()
        pass

    g_jobs.pop(job_id, None)
    return 

def start_training_job(name, from_date, to_date, train_test_split):
    global g_elasticsearch_addr
    global g_pool
    global g_job_id
    global g_jobs

    g_job_id = g_job_id + 1
    args = (g_elasticsearch_addr, name, from_date, to_date)
    g_jobs[g_job_id] = g_pool.apply_async(mp_train_model, args)

    return g_job_id

def start_inference_job(name, from_date, to_date):
    global g_elasticsearch_addr
    global g_pool
    global g_job_id
    global g_jobs

    g_job_id = g_job_id + 1
    args = (g_elasticsearch_addr, name, from_date, to_date)
    g_jobs[g_job_id] = g_pool.apply_async(range_predict, args)

    return g_job_id

def run_nnsom_training_job(name,
                           from_date,
                           to_date,
                           num_epochs=100,
                           ):
    global g_elasticsearch_addr
    global g_pool
    global g_job_id
    global g_jobs

    g_job_id = g_job_id + 1
    args = (g_elasticsearch_addr, name, from_date, to_date, num_epochs)
    g_jobs[g_job_id] = g_pool.apply_async(nnsom_train_model, args)

    return g_job_id

def run_async_map_account(name,
                           from_date,
                           to_date,
                           account_name):
    global g_elasticsearch_addr
    global g_pool
    global g_job_id
    global g_jobs

    g_job_id = g_job_id + 1
    args = (g_elasticsearch_addr, name, account_name, from_date, to_date)
    g_jobs[g_job_id] = g_pool.apply_async(async_map_account, args)

    return g_job_id

def get_job_status(job_id, timeout=1):
    global g_jobs
    res = g_jobs[job_id]
    try:
        successful = res.successful()
    except (AssertionError):
        successful = None

    try:
        result = res.get(timeout)
    except(TimeoutError):
        result = None

    return {
        'ready': res.ready(),
        'successful': successful,
        'result': result, 
    }

def start_predict_job(name, anomaly_threshold=30):
    global g_processes
    global g_elasticsearch_addr

    args = (g_elasticsearch_addr, name, anomaly_threshold)
    p = multiprocessing.Process(target=rt_predict, args=args)
    p.start()
    g_processes[name] = p
    return 

def stop_predict_job(name):
    global g_processes
    p = g_processes[name]
    if p is not None:
        del g_processes[name]
        os.kill(p.pid, signal.SIGUSR1)
        os.waitpid(p.pid, 0)
        return 

def start_nnsom_job(name, anomaly_threshold=30):
    global g_processes
    global g_elasticsearch_addr

    args = (g_elasticsearch_addr, name, anomaly_threshold)
    p = multiprocessing.Process(target=nnsom_rt_predict, args=args)
    p.start()
    g_processes[name] = p
    return 

def stop_nnsom_job(name):
    return stop_predict_job(name)

@app.route('/api/model/create', methods=['POST'])
def create_model():
    global arg
    storage = get_storage()
    # The model name 
    name = request.args.get('name', None)
    # The index name to query at periodic interval 
    index = request.args.get('index', None)
    # ES _routing information to query the index 
    routing = request.args.get('routing', None)
    # time offset, in seconds, when querying the index 
    offset = int(request.args.get('offset', 30))
    # time span, in seconds, to aggregate features
    span = int(request.args.get('span', 300))
    # bucket time span, in seconds, to aggregate features
    bucket_interval = int(request.args.get('bucket_interval', 60))
    # periodic interval to run queries
    interval = int(request.args.get('interval', 60))
    if name is None or index is None:
        return error_msg(msg='Bad Request', code=400)

    # features { .name, .field, .script, .metric }
    data = request.get_json()
    features = data['features']
    if features is None or len(features) == 0:
        return error_msg(msg='Bad Request', code=400)

    storage.create_model(
        name=name,
        index=index,
        routing=routing,
        offset=offset,
        span=span,
        bucket_interval=bucket_interval,
        interval=interval,
        features=features,
    )
  
    return error_msg(msg='', code=200)

@app.route('/api/model/delete', methods=['POST'])
def delete_model():
    storage = get_storage()
    # The model name 
    name = request.args.get('name', None)
    if name is None:
        return error_msg(msg='Bad Request', code=400)

    storage.delete_model(
        name=name,
    )
    return error_msg(msg='', code=200)

# Custom API to create quadrant data based on SUNSHINE paper.
@app.route('/api/nnsom/create', methods=['POST'])
def nnsom_create():
    global arg
    storage = get_storage()
    # The model name 
    name = request.args.get('name', None)
    # The index name to query at periodic interval 
    index = request.args.get('index', None)
    # The term used to aggregate profile data
    term = request.args.get('term', None)
    # The # terms in profile data
    max_terms = int(request.args.get('max_terms', 1000))
    # ES _routing information to query the index 
    routing = request.args.get('routing', None)
    # time offset, in seconds, when querying the index 
    offset = int(request.args.get('offset', 30))
    map_w = int(request.args.get('map_w', 50))
    map_h = int(request.args.get('map_h', 50))

    # periodic interval to run queries
    interval = int(request.args.get('interval', 60))
    if name is None or index is None or term is None:
        return error_msg(msg='Bad Request', code=400)

    storage.create_nnsom(
        name=name,
        index=index,
        routing=routing,
        offset=offset,
        interval=interval,
        term=term,
        max_terms=max_terms,
        map_w=map_w,
        map_h=map_h,
    )
  
    return error_msg(msg='', code=200)

@app.route('/api/nnsom/delete', methods=['POST'])
def nnsom_delete():
    return delete_model()

@app.route('/api/nnsom/get_job_status', methods=['GET'])
def nnsom_job_status():
    return job_status()

@app.route('/api/nnsom/train', methods=['POST'])
def __nnsom_train_model():
    storage = get_storage()
    # The model name 
    name = request.args.get('name', None)
    if name is None:
        return error_msg(msg='Bad Request', code=400)

    from_date = int(request.args.get('from_date', (get_current_time()-30*24*3600)))
    to_date = int(request.args.get('to_date', get_current_time()))
    num_epochs = int(request.args.get('epochs', 100))

    job_id = run_nnsom_training_job(name,
                                    from_date=from_date,
                                    to_date=to_date,
                                    num_epochs=num_epochs,
                                    )

    return jsonify({'job_id': job_id})

@app.route('/api/nnsom/map', methods=['POST'])
def nnsom_map():
    storage = get_storage()
    # The model name 
    name = request.args.get('name', None)
    if name is None:
        return error_msg(msg='Bad Request', code=400)

    account_name = request.args.get('account')
    if account_name is None:
        return error_msg(msg='Bad Request', code=400)

    # By default: calculate the short term (7 days) signature
    from_date = int(request.args.get('from_date', (get_current_time()-7*24*3600)))
    to_date = int(request.args.get('to_date', get_current_time()))

    job_id = run_async_map_account(name,
                                   from_date=from_date,
                                   to_date=to_date,
                                   account_name=account_name,
                                  )

    return jsonify({'job_id': job_id})
    
@app.route('/api/nnsom/start', methods=['POST'])
def nnsom_start_model():
    storage = get_storage()
    # The model name 
    name = request.args.get('name', None)
    if name is None:
        return error_msg(msg='Bad Request', code=400)

    res = storage.find_model(
        name,
    )
    if res == True:
        start_nnsom_job(name)
        return error_msg(msg='', code=200)
    else:
        return error_msg(msg='Not found', code=404)

@app.route('/api/nnsom/stop', methods=['POST'])
def nnsom_stop_model():
    storage = get_storage()
    # The model name 
    name = request.args.get('name', None)
    if name is None:
        return error_msg(msg='Bad Request', code=400)

    res = storage.find_model(
        name,
    )
    if res == True:
        stop_nnsom_job(name)
        return error_msg(msg='', code=200)
    else:
        return error_msg(msg='Not found', code=404)

@app.route('/api/model/get_job_status', methods=['GET'])
def job_status():
    job_id = request.args.get('job_id', None)
    if job_id is None:
        return error_msg(msg='Bad Request', code=400)

    res = get_job_status(int(job_id))
    return make_response(json.dumps(res))

@app.route('/api/model/train', methods=['POST'])
def train_model():
    storage = get_storage()
    # The model name 
    name = request.args.get('name', None)
    if name is None:
        return error_msg(msg='Bad Request', code=400)

    from_date = int(request.args.get('from_date', (get_current_time()-24*3600)))
    to_date = int(request.args.get('to_date', get_current_time()))
    train_test_split = float(request.args.get('train_test_split', 0.67))

    job_id = start_training_job(name, from_date, to_date, train_test_split)

    return jsonify({'job_id': job_id})

@app.route('/api/model/inference', methods=['POST'])
def timeseries_inference():
    storage = get_storage()
    # The model name 
    name = request.args.get('name', None)
    if name is None:
        return error_msg(msg='Bad Request', code=400)

    from_date = int(request.args.get('from_date', (get_current_time()-24*3600)))
    to_date = int(request.args.get('to_date', get_current_time()))

    job_id = start_inference_job(name, from_date, to_date)

    return jsonify({'job_id': job_id})

    
@app.route('/api/model/start', methods=['POST'])
def start_model():
    storage = get_storage()
    # The model name 
    name = request.args.get('name', None)
    if name is None:
        return error_msg(msg='Bad Request', code=400)

    res = storage.find_model(
        name,
    )
    if res == True:
        start_predict_job(name)
        return error_msg(msg='', code=200)
    else:
        return error_msg(msg='Not found', code=404)

@app.route('/api/model/stop', methods=['POST'])
def stop_model():
    storage = get_storage()
    # The model name 
    name = request.args.get('name', None)
    if name is None:
        return error_msg(msg='Bad Request', code=400)

    res = storage.find_model(
        name,
    )
    if res == True:
        stop_predict_job(name)
        return error_msg(msg='', code=200)
    else:
        return error_msg(msg='Not found', code=404)

@app.route('/api/model/list', methods=['GET'])
def list_models():
    storage = get_storage()
    size = int(request.args.get('size', 10))

    return jsonify(storage.get_model_list(
            size=size,
        ))

def main():
    global g_elasticsearch_addr
    global g_pool
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
        '-l', '--listen',
        help="Listen address",
        type=str,
        default="0.0.0.0:8077",
    )
    parser.add_argument(
        '--maxtasksperchild',
        help="Maxtasksperchild in process pool size",
        type=int,
        default=10,
    )
    parser.add_argument(
        '-w', '--workers',
        help="Worker processes pool size",
        type=int,
        default=multiprocessing.cpu_count(),
    )

    arg = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    app.logger.setLevel(logging.INFO)

    g_elasticsearch_addr = arg.elasticsearch_addr

    storage = get_storage()
    try:
        es_res = storage.get_model_list()
        for doc in es_res:
            try:
                if 'term' in doc:
                    continue # start_nnsom_job(doc['name'])
                else:
                    start_predict_job(doc['name'])
            except(Exception):
                pass

    except(StorageException):
        pass
   
    g_pool = Pool(processes=arg.workers, maxtasksperchild=arg.maxtasksperchild)
 
    host, port = arg.listen.split(':')
    app.run(host=host, port=port)

    g_pool.close()
    g_pool.join()

if __name__ == "__main__":
    # execute only if run as a script
    main()


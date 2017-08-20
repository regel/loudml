import argparse
import datetime
import time
import base64
import logging
import json
import os
import sys

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

arg = None
threadLocal = threading.local()

app = Flask(__name__, static_url_path='/static', template_folder='templates')

def get_storage():
    storage = getattr(threadLocal, 'storage', None)
    if storage is None:
        storage = Storage(app.config['ES_ADDR'])
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


def terminate_pending_jobs(name):
    # FIXME
    return 

def start_training_job(name, from_date, to_date, train_test_split):
    # FIXME
    return 

def start_predict_job(name):
    # FIXME
    return 

def stop_predict_job(name):
    # FIXME
    return 



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

    terminate_pending_jobs(name)

    storage.delete_model(
        name=name,
    )
    return error_msg(msg='', code=200)

@app.route('/api/model/train', methods=['POST'])
def train_model():
    storage = get_storage()
    # The model name 
    name = request.args.get('name', None)
    if name is None:
        return error_msg(msg='Bad Request', code=400)

    from_date = request.args.get('from_date', 1000 * (get_current_time()-24*3600))
    to_date = request.args.get('to_date', 1000 * get_current_time())
    train_test_split = float(request.args.get('train_test_split', 0.67))

    start_training_job(name, from_date, to_date, train_test_split)

    return error_msg(msg='Not found', code=404)
    
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
        return start_predict_job(name)
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
        return stop_predict_job(name)
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

    arg = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    app.logger.setLevel(logging.INFO)

    app.config['ES_ADDR'] = arg.elasticsearch_addr

    storage = get_storage()
    try:
        es_res = storage.get_model_list()
        for doc in es_res:
            try:
                start_predict_job(doc['name'])
            except(Exception):
                pass
    except(StorageException):
        pass
    
    host, port = arg.listen.split(':')
    app.run(host=host, port=port)

if __name__ == "__main__":
    # execute only if run as a script
    main()


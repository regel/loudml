"""
LoudML server
"""

import loudml.vendor

import argparse
import datetime
import logging
import multiprocessing
import multiprocessing.pool
import pkg_resources
import queue
import sys
import uuid

import fcntl
import psutil
import inspect

import loudml.config
import loudml.model
import loudml.worker

from threading import (
    Timer,
)

from flask import (
    Flask,
    jsonify,
    request,
)
from flask_restful import (
    Api,
    Resource,
)
from gevent.pywsgi import (
    WSGIServer,
)
from . import (
    errors,
    schemas,
)
from .filestorage import (
    FileStorage,
)
from .misc import (
    make_bool,
    load_entry_point,
)

app = Flask(__name__, static_url_path='/static', template_folder='templates')
api = Api(app)

g_config = None
g_jobs = {}
g_training = {}
g_storage = None
g_pool = None
g_queue = None
g_running_models = {}

MAX_RUNNING_MODELS = 3

# Do not change: pid file to ensure we're running single instance
APP_INSTALL_PATHS = [
    "/usr/bin/loudmld",
    "/bin/loudmld", # With RPM, binaries are also installed here
]
LOCK_FILE = "/var/tmp/loudmld.lock"

class RepeatingTimer(object):
    def __init__(self, interval, cb, *args, **kwargs):
        self.interval = interval
        self.cb = cb
        self.args = args
        self.kwargs = kwargs
        self.timer = None

    def callback(self):
        self.cb(*self.args, **self.kwargs)
        self.start()

    def cancel(self):
        self.timer.cancel()

    def start(self):
        self.timer = Timer(self.interval, self.callback)
        self.timer.start()


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class Pool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


class Job:
    """
    LoudML job
    """
    func = None
    job_type = None

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.state = 'idle'
        self.result = None
        self.error = None
        self.progress = None

    @property
    def desc(self):
        desc = {
            'id': self.id,
            'type': self.job_type,
            'state': self.state,
        }
        if self.result:
            desc['result'] = self.result
        if self.error:
            desc['error'] = self.error
        if self.progress:
            desc['progress'] = self.progress
        return desc

    @property
    def args(self):
        return []

    @property
    def kwargs(self):
        return {}

    def start(self):
        """
        Submit job to worker pool
        """
        global g_pool
        global g_jobs

        g_jobs[self.id] = self
        self.state = 'waiting'
        g_pool.apply_async(
            func=loudml.worker.run,
            args=[self.id, self.func] + self.args,
            kwds=self.kwargs,
            callback=self.done,
            error_callback=self.fail,
        )

    def done(self, result):
        """
        Callback executed when job is done
        """
        self.state = 'done'
        self.result = result
        logging.info("job[%s] done", self.id)

    def fail(self, error):
        """
        Callback executed when job fails
        """
        self.state = 'failed'
        self.error = str(error)
        logging.info("job[%s] failed: %s", self.id, self.error)


def set_job_state(job_id, state, progress=None):
    """
    Set job state
    """
    global g_jobs

    job = g_jobs.get(job_id)

    if job is None:
        logging.warning("got message for unknown job '%s'", job_id)
        return

    if job.state in ['done', 'failed']:
        # Too late
        return

    job.state = state
    job.progress = progress

def read_messages():
    """
    Read messages from subprocesses
    """
    global g_queue

    while True:
        try:
            msg = g_queue.get(block=False)

            if msg['type'] == 'job_state':
                set_job_state(
                    msg['job_id'],
                    msg['state'],
                    progress=msg.get('progress'),
                )
        except queue.Empty:
            break

@app.errorhandler(errors.Invalid)
def handle_loudml_error(exn):
    response = jsonify({
        'error': str(exn),
    })
    response.status_code = exn.code
    return response

def catch_loudml_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except errors.LoudMLException as exn:
            return str(exn), exn.code

    wrapper.__name__ = '{}_wrapper'.format(func.__name__)
    return wrapper

def get_bool_arg(param, default=False):
    """
    Read boolean URL parameter
    """
    try:
        return make_bool(request.args.get(param, default))
    except ValueError:
        raise errors.Invalid("invalid value for parameter '{}'".format(param))

def get_int_arg(param, default=None):
    """
    Read integer URL parameter
    """
    try:
        return int(request.args[param])
    except KeyError:
        return default
    except ValueError:
        raise errors.Invalid("invalid value for parameter '{}'".format(param))

def get_date_arg(param, default=None, is_mandatory=False):
    """
    Read date URL parameter
    """
    try:
        value = request.args[param]
    except KeyError:
        if is_mandatory:
            raise errors.Invalid("'{}' parameter is required".format(param))
        return default

    return schemas.validate(schemas.Timestamp(), value, name=param)

class ModelsResource(Resource):
    @catch_loudml_error
    def get(self):
        global g_storage

        models = []

        for name in g_storage.list_models():
            model = g_storage.load_model(name)
            models.append({
                'settings': model.settings,
            })

        return jsonify(models)

    @catch_loudml_error
    def put(self):
        global g_storage

        model = loudml.model.load_model(settings=request.json)
        g_storage.create_model(model)

        return "success", 201


class ModelResource(Resource):
    @catch_loudml_error
    def get(self, model_name):
        global g_storage

        model = g_storage.load_model(model_name)

        preview = model.preview

        return jsonify(model.preview)

    @catch_loudml_error
    def delete(self, model_name):
        global g_storage

        g_storage.delete_model(model_name)
        logging.info("model '%s' deleted", model_name)
        return "success"

    @catch_loudml_error
    def post(self, model_name):
        global g_storage

        settings = request.json

        if settings is None:
            return "model description is missing", 400

        settings['name'] = model_name
        model = loudml.model.load_model(settings=settings)

        try:
            g_storage.delete_model(model_name)
        except errors.ModelNotFound:
            pass

        g_storage.create_model(model)
        logging.info("model '%s' updated", model_name)
        return "success"


api.add_resource(ModelsResource, "/models")
api.add_resource(ModelResource, "/models/<model_name>")

@app.route("/models/<model_name>/_train", methods=['POST'])
def model_train(model_name):
    global g_storage
    global g_training

    model = g_storage.load_model(model_name)
    kwargs = {}

    kwargs['from_date'] = get_date_arg('from', is_mandatory=True)
    kwargs['to_date'] = get_date_arg('to')

    datasource = request.args.get('datasource')
    if datasource is not None:
        kwargs['datasource'] = datasource

    max_evals = get_int_arg('max_evals')
    if max_evals is not None:
        kwargs['max_evals'] = max_evals

    job = TrainingJob(model_name, **kwargs)
    job.start()

    g_training[model_name] = job

    return jsonify(job.id)

class HooksResource(Resource):
    @catch_loudml_error
    def get(self, model_name):
        global g_storage

        return jsonify(g_storage.list_model_hooks(model_name))

    @catch_loudml_error
    def put(self, model_name):
        global g_storage

        data = request.json
        if data is None:
            return "hook data is missing", 400

        hook_type = data.get('type')
        if hook_type is None:
            return "type is missing", 400

        hook_name = data.get('name')
        if hook_name is None:
            return "name is missing", 400

        hook = load_entry_point('loudml.hooks', hook_type)
        if hook is None:
            return "unknown hook type", 404

        config = data.get('config')
        hook.validate(config)
        g_storage.set_model_hook(model_name, hook_name, hook_type, config)

        return "success", 201


class HookResource(Resource):
    @catch_loudml_error
    def get(self, model_name, hook_name):
        global g_storage

        hook = g_storage.get_model_hook(model_name, hook_name)
        return jsonify(hook)

    @catch_loudml_error
    def delete(self, model_name, hook_name):
        global g_storage

        g_storage.delete_model_hook(model_name, hook_name)
        logging.info("hook '%s/%s' deleted", model_name, hook_name)
        return "success"

    @catch_loudml_error
    def post(self, model_name):
        global g_storage

        config = request.json

        if settings is None:
            return "model description is missing", 400

        data = request.json
        hook_type = request.args.get('type')
        if hook_type is None:
            return "type is missing", 400

        hook = load_entry_point('loudml.hook', hook_type)
        hook.validate(data.get('config'))
        g_storage.set_model_hook(hook_name, hook_type, config)

        logging.info("hook '%s/%s' updated", model_name, hook_name)
        return "success"


api.add_resource(HooksResource, "/models/<model_name>/hooks")
api.add_resource(HookResource, "/models/<model_name>/hooks/<hook_name>")

@app.route("/models/<model_name>/hooks/<hook_name>/_test", methods=['POST'])
@catch_loudml_error
def hook_test(model_name, hook_name):
    global g_storage

    model = g_storage.load_model(model_name)
    hook = g_storage.load_model_hook(model_name, hook_name)

    model.load()
    prediction = model.generate_fake_prediction()
    model.detect_anomalies(prediction, [hook])

    return "ok", 200

class DataSourcesResource(Resource):
    @catch_loudml_error
    def get(self):
        return jsonify(list(g_config.datasources.values()))


class DataSourceResource(Resource):
    @catch_loudml_error
    def get(self, datasource_name):
        datasource = g_config.get_datasource(datasource_name)
        return jsonify(datasource)


api.add_resource(DataSourcesResource, "/datasources")
api.add_resource(DataSourceResource, "/datasources/<datasource_name>")

class JobsResource(Resource):
    @catch_loudml_error
    def get(self):
        global g_jobs
        return jsonify([job.desc for job in g_jobs.values()])


class JobResource(Resource):
    @catch_loudml_error
    def get(self, job_id):
        global g_jobs

        job = g_jobs.get(job_id)
        if job is None:
            return "job not found", 404

        return jsonify(job.desc)


api.add_resource(JobsResource, "/jobs")
api.add_resource(JobResource, "/jobs/<job_id>")

class TrainingJob(Job):
    """
    Model training job
    """
    func = 'train'
    job_type = 'training'

    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.model_name = model_name
        self._kwargs = kwargs

    @property
    def args(self):
        return [self.model_name]

    @property
    def kwargs(self):
        return self._kwargs


@app.route("/models/<model_name>/training")
def model_training_job(model_name):
    global g_training

    job = g_training.get(model_name)
    if job is None:
        return "training job not found", 404

    return jsonify(job.desc)

class PredictionJob(Job):
    """
    Prediction job
    """
    func = 'predict'
    job_type = 'prediction'

    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.model_name = model_name
        self._kwargs = kwargs

    @property
    def args(self):
        return [self.model_name]

    @property
    def kwargs(self):
        return self._kwargs

class ForecastJob(Job):
    """
    Forecast job
    """
    func = 'forecast'
    job_type = 'forecast'

    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.model_name = model_name
        self._kwargs = kwargs

    @property
    def args(self):
        return [self.model_name]

    @property
    def kwargs(self):
        return self._kwargs

def _model_start(model, params):
    """
    Start periodic prediction
    """

    if model.name in g_running_models:
        raise errors.Conflict(
            "real-time prediction is already active for this model",
        )

    if len(g_running_models) >= MAX_RUNNING_MODELS:
        raise errors.LimitReached(
            "maximum number of running models is reached",
        )

    def create_job(from_date=None):
        kwargs = params.copy()

        if model.type == 'timeseries':
            to_date = datetime.datetime.now().timestamp() - model.offset

            if from_date is None:
                from_date = to_date - model.bucket_interval

            kwargs['from_date'] = from_date
            kwargs['to_date'] = to_date
        elif model.type.endswith('fingerprints'):
            to_date = datetime.datetime.now().timestamp() - model.offset

            if from_date is None:
                from_date = to_date - model.span

            kwargs['from_date'] = from_date
            kwargs['to_date'] = to_date

        job = PredictionJob(model.name, **kwargs)
        job.start()

    from_date = params.pop('from', None)
    create_job(from_date)

    timer = RepeatingTimer(model.interval, create_job)
    g_running_models[model.name] = timer
    timer.start()

def _model_forecast(model, params):
    """
    Run forecast (one-shot)
    """
    job = ForecastJob(model.name, **params)
    job.start()
    return str(job.id)

@app.route("/models/<model_name>/_start", methods=['POST'])
@catch_loudml_error
def model_start(model_name):
    global g_storage

    params = {
        'save_prediction': get_bool_arg('save_prediction'),
        'detect_anomalies': get_bool_arg('detect_anomalies'),
    }

    model = g_storage.load_model(model_name)
    model.set_run_params(params)
    g_storage.save_model(model)

    params['from'] = get_date_arg('from')
    _model_start(model, params)

    return "real-time prediction started", 200

@app.route("/models/<model_name>/_stop", methods=['POST'])
@catch_loudml_error
def model_stop(model_name):
    global g_running_models
    global g_storage

    timer = g_running_models.get(model_name)
    if timer is None:
        return "model is not active", 404

    timer.cancel()
    del g_running_models[model_name]
    logging.info("model '%s' deactivated", model_name)

    model = g_storage.load_model(model_name)
    model.set_run_params(None)
    g_storage.save_model(model)

    return "model deactivated"

@app.route("/models/<model_name>/_forecast", methods=['POST'])
@catch_loudml_error
def model_forecast(model_name):
    global g_storage

    params = {
        'save_prediction': get_bool_arg('save_prediction'),
    }

    model = g_storage.load_model(model_name)

    params['from_date'] = get_date_arg('from', default='now')
    params['to_date'] = get_date_arg('to', defaut=None)
    job_id = _model_forecast(model, params)

    return job_id, 200

#
# Example of job
#
#class DummyJob(Job):
#    func = 'do_things'
#    job_type = 'dummy'
#
#    def __init__(self, value):
#        super().__init__()
#        self.value = value
#
#    @property
#    def args(self):
#        return [self.value]
#
# Example of endpoint that submits jobs
#@app.route("/do-things")
#def do_things():
#    job = DummyJob(int(request.args.get('value', 0)))
#    job.start()
#    return str(job.id)

@app.route("/")
def slash():
    version = pkg_resources.get_distribution("loudml").version
    return jsonify({
        'version': version,
        'tagline': "The Disruptive Machine Learning API",
    })

@app.errorhandler(403)
def not_found(e):
    return "forbidden", 403

@app.errorhandler(404)
def not_found(e):
    return "unknown endpoint", 404

@app.errorhandler(405)
def not_found(e):
    return "method not allowed", 405

@app.errorhandler(410)
def not_found(e):
    return "gone", 410

@app.errorhandler(500)
def not_found(e):
    return "internal server error", 500

def check_instance():
    stack_data = inspect.stack()
    app_path = stack_data[-1][0].f_code.co_filename
    if app_path not in APP_INSTALL_PATHS:
        logging.error("Unauthorized instance")
        sys.exit(1)

    fp = open(LOCK_FILE, 'w')
    try:
        fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        running = 0
        for p in psutil.process_iter():
            cmd = p.cmdline()
            if len(cmd) > 1 and cmd[1] in APP_INSTALL_PATHS:
                running = running + 1

        if running > 1:
            logging.error("Another instance running")
            sys.exit(1)

    except IOError as exn:
        logging.error("Another instance running")
        sys.exit(1)

def restart_predict_jobs():
    """
    Restart prediction jobs
    """

    global g_storage

    for name in g_storage.list_models():
        model = g_storage.load_model(name)

        params = model.settings.get('run')
        if params is None:
            continue

        try:
            logging.info("restarting job for model '%s'", model.name)
            _model_start(model, params)
        except errors.LoudMLException as exn:
            logging.error("cannot restart job for model '%s'", model.name)

def main():
    """
    LoudML server
    """

    global g_config
    global g_pool
    global g_queue
    global g_storage

    parser = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-c', '--config',
        help="Path to configuration file",
        type=str,
        default="/etc/loudml/config.yml",
    )

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    app.logger.setLevel(logging.INFO)

    #DISABLED check_instance()

    try:
        g_config = loudml.config.load_config(args.config)
        g_storage = FileStorage(g_config.storage['path'])
        loudml.config.load_plugins(args.config)
    except errors.LoudMLException as exn:
        logging.error(exn)
        sys.exit(1)

    g_queue = multiprocessing.Queue()
    g_pool = Pool(
        processes=g_config.server['workers'],
        initializer=loudml.worker.init_worker,
        initargs=[args.config, g_queue],
        maxtasksperchild=g_config.server['maxtasksperchild'],
    )

    timer = RepeatingTimer(1, read_messages)
    timer.start()

    host, port = g_config.server['listen'].split(':')

    restart_predict_jobs()

    try:
        http_server = WSGIServer((host, int(port)), app)
        http_server.serve_forever()
    except KeyboardInterrupt:
        pass

    logging.info("stopping")
    timer.cancel()
    g_pool.close()
    g_pool.join()

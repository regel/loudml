"""
LoudML server
"""

import loudml.vendor

import argparse
import concurrent.futures
import datetime
import logging
import multiprocessing
import pebble
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
    parse_timedelta,
    parse_constraint,
)
from .license import (
    License,
)

app = Flask(__name__, static_url_path='/static', template_folder='templates')
api = Api(app)

g_lock = multiprocessing.Lock()
g_config = None
g_jobs = {}
g_training = {}
g_storage = None
g_pool = None
g_queue = None
g_running_models = {}

# Do not change: pid file to ensure we're running single instance
APP_INSTALL_PATHS = [
    "/usr/bin/loudmld",
    "/bin/loudmld",  # With RPM, binaries are also installed here
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


class Job:
    """
    LoudML job
    """
    func = None
    job_type = None

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.state = 'idle'
        self._result = None
        self.error = None
        self.progress = None
        self._future = None

    @property
    def desc(self):
        desc = {
            'id': self.id,
            'type': self.job_type,
            'state': self.state,
        }
        if self.result:
            desc['result'] = self._result
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

    def is_stopped(self):
        """
        Tell if job is stopped
        """
        return self.state in ['done', 'failed', 'canceled']

    def start(self):
        """
        Submit job to worker pool
        """
        global g_pool
        global g_jobs

        g_jobs[self.id] = self
        self.state = 'waiting'
        self._future = g_pool.schedule(
            loudml.worker.run,
            args=[self.id, self.func] + self.args,
            kwargs=self.kwargs,
        )
        self._future.add_done_callback(self._done_cb)

    def cancel(self):
        """
        Cancel job
        """

        if self.is_stopped():
            raise errors.Conflict(
                "job is already stopped (state = {})".format(self.state),
            )

        self.state = 'canceling'
        logging.info("job[%s] canceling...", self.id)
        self._future.cancel()

    def _done_cb(self, result):
        """
        Callback executed when job is done
        """

        try:
            self._result = self._future.result()
            self.state = 'done'
            logging.info("job[%s] done", self.id)
        except concurrent.futures.CancelledError as exn:
            self.error = "job canceled"
            self.state = 'canceled'
            logging.error("job[%s] canceled", self.id)
        except Exception as exn:
            self.error = str(exn)
            self.state = 'failed'
            logging.error("job[%s] failed: %s", self.id, self.error)

    def result(self):
        """
        Return job result
        """
        return self._future.result()


@app.route("/jobs/<job_id>/_cancel", methods=['POST'])
def job_stop(job_id):
    global g_jobs

    job = g_jobs.get(job_id)
    if job is None:
        return "job not found", 404

    job.cancel()
    return "job canceled"


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


@app.errorhandler(errors.LoudMLException)
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


def get_model_info(name):
    global g_storage
    global g_training

    model = g_storage.load_model(name)
    info = model.preview

    job = g_training.get(name)
    if job:
        job_desc = job.desc

        training = {
            'job_id': job.id,
            'state': job_desc['state'],
        }
        progress = job_desc.get('progress')
        if progress:
            training['progress'] = progress

        info['training'] = training

    return info


class ModelsResource(Resource):
    @catch_loudml_error
    def get(self):
        models = []

        for name in g_storage.list_models():
            try:
                models.append(get_model_info(name))
            except errors.UnsupportedModel:
                continue

        return jsonify(models)

    @catch_loudml_error
    def put(self):
        global g_config
        global g_storage

        settings = request.json
        model = loudml.model.load_model(settings=settings, config=g_config)

        g_storage.create_model(model, g_config)

        return "success", 201


class ModelResource(Resource):
    @catch_loudml_error
    def get(self, model_name):
        return jsonify(get_model_info(model_name))

    @catch_loudml_error
    def delete(self, model_name):
        global g_storage
        global g_running_models
        global g_training

        g_lock.acquire()

        try:
            timer = g_running_models.pop(model_name, None)
            if timer:
                timer.cancel()

            job = g_training.get(model_name)
            if job and job.is_stopped() == False:
                job.cancel()

            g_storage.delete_model(model_name)
        finally:
            g_lock.release()

        logging.info("model '%s' deleted", model_name)
        return "success"

    @catch_loudml_error
    def post(self, model_name):
        global g_config
        global g_storage
        global g_running_models

        settings = request.json

        if settings is None:
            return "model description is missing", 400

        settings['name'] = model_name
        model = loudml.model.load_model(settings=settings, config=g_config)

        changes = g_storage.save_model(model, save_state=False)
        for change, param, desc in changes: 
            logging.info("model '%s' %s %s %s", model_name, change, param, desc)
            if change == 'change' and param == 'interval':
                previous_val, next_val = desc
                g_lock.acquire() 
                timer = g_running_models.get(model_name)
                if timer is not None:
                    timer.cancel()
                    timer.interval = parse_timedelta(next_val).total_seconds()
                    timer.start()
                g_lock.release() 

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
    kwargs['to_date'] = get_date_arg('to', default="now")
    if get_bool_arg('autostart'):
        kwargs.update({
            'autostart': True,
            'save_prediction': get_bool_arg('save_prediction'),
            'datasink': request.args.get('datasink'),
            'detect_anomalies': get_bool_arg('detect_anomalies'),
        })

    datasource = request.args.get('datasource')
    if datasource is not None:
        kwargs['datasource'] = datasource

    max_evals = get_int_arg('max_evals')
    if max_evals is not None:
        kwargs['max_evals'] = max_evals

    kwargs['license'] = g_config.license

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
    def post(self, model_name, hook_name):
        global g_storage

        if model_name is None:
            return "model description is missing", 400

        data = request.json
        if data is None:
            return "hook description is missing", 400

        hook_type = data.get('type')
        if hook_type is None:
            return "type is missing", 400

        hook = load_entry_point('loudml.hooks', hook_type)
        if hook is None:
            return "unknown hook type", 404

        config = data.get('config')
        hook.validate(config)
        g_storage.set_model_hook(model_name, hook_name, hook_type, config)

        logging.info("hook '%s/%s' updated", model_name, hook_name)
        return "success"


api.add_resource(HooksResource, "/models/<model_name>/hooks")
api.add_resource(HookResource, "/models/<model_name>/hooks/<hook_name>")


@app.route("/models/<model_name>/hooks/<hook_name>/_test", methods=['POST'])
@catch_loudml_error
def hook_test(model_name, hook_name):
    global g_storage

    model = g_storage.load_model(model_name)
    hook = g_storage.load_model_hook(model.settings, hook_name)

    model.load()
    prediction = model.generate_fake_prediction()
    model.detect_anomalies(prediction, [hook])

    return "ok", 200

def _remove_datasource_secrets(datasource):
    datasource.pop('password', None)
    datasource.pop('dbuser_password', None)
    datasource.pop('write_token', None)
    datasource.pop('read_token', None)

class DataSourcesResource(Resource):
    @catch_loudml_error
    def get(self):
        res = []
        for datasource in g_config.datasources.values():
            _remove_datasource_secrets(datasource)
            res.append(datasource)
        return jsonify(res)


class DataSourceResource(Resource):
    @catch_loudml_error
    def get(self, datasource_name):
        datasource = g_config.get_datasource(datasource_name)
        _remove_datasource_secrets(datasource)
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
        self.autostart = kwargs.pop('autostart', False)
        self._kwargs_start = {
            'save_prediction': kwargs.pop('save_prediction', False),
            'datasink': kwargs.pop('datasink', None),
            'detect_anomalies': kwargs.pop('detect_anomalies', False),
            'from_date': kwargs.get('from_date', None),
        }
        self._kwargs = kwargs

    def _done_cb(self, result):
        """
        Callback executed when job is done
        """
        super()._done_cb(result)
        if self.state == 'done' and self.autostart:
            logging.info("scheduling autostart for model '%s'", self.model_name)
            model = g_storage.load_model(self.model_name)
            params = self._kwargs_start.copy()
            params.pop('from_date')
            model.set_run_params(params)
            g_storage.save_model(model)
            try:
                _model_start(model, self._kwargs_start)
            except errors.LoudMLException as exn:
                model.set_run_params(None)
                g_storage.save_model(model)


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
    g_lock.acquire()
    if model.name in g_running_models:
        g_lock.release()
        # nothing to do; the job will load the model file
        return

    if len(g_running_models) >= g_config.license.max_running_models:
        g_lock.release()
        raise errors.LimitReached(
            "You've reached the maximum count allowed in your license",
        )

    def create_job(from_date=None, save_run_state=True, detect_anomalies=None):
        kwargs = params.copy()
        if detect_anomalies is not None:
            kwargs['detect_anomalies'] = detect_anomalies

        to_date = datetime.datetime.now().timestamp() - model.offset

        if model.type == 'timeseries':
            if from_date is None:
                from_date = to_date - model.interval

            kwargs['from_date'] = from_date
            kwargs['to_date'] = to_date
        elif model.type.endswith('fingerprints'):
            if from_date is None:
                from_date = to_date - model.span

            kwargs['from_date'] = from_date
            kwargs['to_date'] = to_date

        job = PredictionJob(
            model.name,
            save_run_state=save_run_state,
            **kwargs
        )

        job.start()

    from_date = params.pop('from_date', None)
    create_job(from_date, save_run_state=False, detect_anomalies=False)

    timer = RepeatingTimer(model.interval, create_job)
    g_running_models[model.name] = timer
    g_lock.release()
    timer.start()


@app.route("/models/<model_name>/_predict", methods=['POST'])
def model_predict(model_name):
    global g_storage
    global g_config

    job = PredictionJob(
        model_name,
        save_run_state=False,
        from_date=get_date_arg('from', is_mandatory=True),
        to_date=get_date_arg('to', is_mandatory=True),
        save_prediction=request.args.get('save_prediction', default=False),
        datasink=request.args.get('datasink'),
        detect_anomalies=request.args.get('detect_anomalies', default=False),
        license= g_config.license
    )
    job.start()

    if get_bool_arg('bg', default=False):
        return str(job.id)

    return jsonify(job.result())


@app.route("/models/<model_name>/_start", methods=['POST'])
def model_start(model_name):
    global g_storage

    params = {
        'save_prediction': get_bool_arg('save_prediction'),
        'datasink': request.args.get('datasink'),
        'detect_anomalies': get_bool_arg('detect_anomalies'),
    }

    model = g_storage.load_model(model_name)
    if not model.is_trained:
        raise errors.ModelNotTrained()

    model.set_run_params(params)
    model.set_run_state(None)
    g_storage.save_model(model)

    params['from_date'] = get_date_arg('from')
    try:
        _model_start(model, params)
    except errors.LoudMLException as exn:
        model.set_run_params(None)
        g_storage.save_model(model)
        raise(exn)

    return "real-time prediction started", 200


@app.route("/models/<model_name>/_stop", methods=['POST'])
@catch_loudml_error
def model_stop(model_name):
    global g_running_models
    global g_storage

    g_lock.acquire()
    timer = g_running_models.get(model_name)
    if timer is None:
        g_lock.release()
        return "model is not active", 404

    timer.cancel()
    del g_running_models[model_name]
    g_lock.release()
    logging.info("model '%s' deactivated", model_name)

    model = g_storage.load_model(model_name)
    model.set_run_params(None)
    model.set_run_state(None)
    g_storage.save_model(model)

    return "model deactivated"


@app.route("/models/<model_name>/_forecast", methods=['POST'])
@catch_loudml_error
def model_forecast(model_name):
    global g_storage

    params = {
        'save_prediction': get_bool_arg('save_prediction'),
        'datasink': request.args.get('datasink'),
    }

    model = g_storage.load_model(model_name)

    params['from_date'] = get_date_arg('from', default='now')
    params['to_date'] = get_date_arg('to', is_mandatory=True)

    constraint = request.args.get('constraint')
    if constraint:
        params['constraint'] = parse_constraint(constraint)

    params['license'] = g_config.license

    job = ForecastJob(model.name, **params)
    job.start()

    if get_bool_arg('bg', default=False):
        return str(job.id)

    return jsonify(job.result())

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


@app.route("/license")
def license():
    return jsonify(g_config.license.payload)


@app.route("/")
def slash():
    version = pkg_resources.get_distribution("loudml").version
    return jsonify({
        'version': version,
        'tagline': "The Disruptive Machine Learning API",
        'host_id': License.my_host_id(),
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

    except IOError:
        logging.error("Another instance running")
        sys.exit(1)


def restart_predict_jobs():
    """
    Restart prediction jobs
    """

    global g_storage

    for name in g_storage.list_models():
        try:
            model = g_storage.load_model(name)
        except errors.LoudMLException as exn:
            logging.error("exception loading model '%s':%s", name, exn)
            continue

        params = model.settings.get('run')
        if params is None:
            continue

        try:
            logging.info("restarting job for model '%s'", model.name)
            _model_start(model, params)
        except errors.LoudMLException:
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
    g_pool = pebble.ProcessPool(
        max_workers=g_config.server.get('workers', 1),
        max_tasks=g_config.server.get('maxtasksperchild', 1),
        initializer=loudml.worker.init_worker,
        initargs=[args.config, g_queue],
    )

    timer = RepeatingTimer(1, read_messages)
    timer.start()

    host, port = g_config.server['listen'].split(':')

    restart_predict_jobs()

    try:
        http_server = WSGIServer((host, int(port)), app)
        http_server.serve_forever()
    except OSError as exn:
        logging.error(str(exn))
    except KeyboardInterrupt:
        pass

    logging.info("stopping")
    timer.cancel()
    g_pool.stop()
    g_pool.join()

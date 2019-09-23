"""
Loud ML server
"""

import loudml.vendor

from crontab import CronTab

import argparse
import concurrent.futures
from datetime import datetime, timedelta
import logging
import multiprocessing
import pebble
import pkg_resources
import queue
import schedule
import sys
import uuid
import traceback
import pytz
import requests

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
    Response,
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
from .bucket import (
    load_bucket,
)
from .filestorage import (
    FileStorage,
)
from .metrics import (
    send_metrics,
)
from .misc import (
    clear_fields,
    make_bool,
    my_host_id,
    load_entry_point,
    parse_timedelta,
    parse_constraint,
    parse_expression,
)

app = Flask(__name__, static_url_path='/static', template_folder='templates')
api = Api(app)

g_config = None
g_jobs = {}
g_scheduled_jobs = {}
g_training = {}
g_storage = None
g_training_pool = None
g_pool = None
g_nice = 0
g_queue = None
g_timer = None

# Do not change: pid file to ensure we're running single instance
APP_INSTALL_PATHS = [
    "/usr/bin/loudmld",
    "/bin/loudmld",  # With RPM, binaries are also installed here
]
LOCK_FILE = "/var/tmp/loudmld.lock"


def get_job_desc(job_id, fields=None, include_fields=None):
    global g_jobs
    desc = g_jobs[job_id].desc
    if fields:
        clear_fields(desc, fields, include_fields)
    return desc


def get_sched_job_desc(job_id, fields=None, include_fields=None):
    global g_scheduled_jobs
    desc = g_scheduled_jobs[job_id]
    if fields:
        clear_fields(desc, fields, include_fields)
    return desc


def get_schedule(cnt, unit, time_str=None):
    unit_map = {
        'second': schedule.every(cnt).second,
        'seconds': schedule.every(cnt).seconds,
        'minute': schedule.every(cnt).minute,
        'minutes': schedule.every(cnt).minutes,
        'hour': schedule.every(cnt).hour,
        'hours': schedule.every(cnt).hours,
        'day': schedule.every(cnt).day,
        'days': schedule.every(cnt).days,
        'week': schedule.every(cnt).week,
        'weeks': schedule.every(cnt).weeks,
        'monday': schedule.every(1).monday,
        'tuesday': schedule.every(1).tuesday,
        'wednesday': schedule.every(1).wednesday,
        'thursday': schedule.every(1).thursday,
        'friday': schedule.every(1).friday,
        'saturday': schedule.every(1).saturday,
        'sunday': schedule.every(1).sunday,
    }
    scheduled_event = unit_map.get(unit)
    if time_str:
        scheduled_event = scheduled_event.at(time_str)
    return scheduled_event


def daemon_exec_scheduled_job(job_id):
    global g_scheduled_jobs
    global g_config

    desc = g_scheduled_jobs[job_id]
    listen_addr = g_config.server['listen']
    host, port = listen_addr.split(':')
    target_url = 'http://{}:{}{}'.format(
        host, port, desc['relative_url'])
    params = desc.get('params')
    if desc['method'] == 'get':
        response = requests.get(target_url, params)
    elif desc['method'] == 'head':
        response = requests.head(target_url, params)
    elif desc['method'] == 'post':
        response = requests.post(
            target_url, params, json=desc.get('json'))
    elif desc['method'] == 'delete':
        response = requests.delete(target_url, params)
    elif desc['method'] == 'patch':
        response = requests.patch(
            target_url, params, json=desc.get('json'))

    desc['ok'] = response.ok
    desc['status_code'] = response.status_code
    desc['error'] = response.reason
    desc['last_run_timestamp'] = datetime.now(pytz.utc).timestamp()
    if not response.ok:
        logging.error(
            "error executing scheduled job '%s':%s",
            desc['name'],
            response.reason)


def add_new_scheduled_job(desc):
    global g_scheduled_jobs
    scheduled_job_name = desc['name']
    scheduled_event = get_schedule(
        cnt=desc['every'].get('count', 1),
        unit=desc['every']['unit'],
        time_str=desc['every'].get('at'))

    g_scheduled_jobs[scheduled_job_name] = desc
    scheduled_event.do(
        daemon_exec_scheduled_job, scheduled_job_name).tag(
        'scheduled_job:{}'.format(scheduled_job_name),
        'scheduled_job')
    return scheduled_job_name


def scheduled_job_exists(scheduled_job_name):
    global g_scheduled_jobs
    return scheduled_job_name in g_scheduled_jobs


def del_scheduled_job(scheduled_job_name):
    global g_scheduled_jobs
    if scheduled_job_name in g_scheduled_jobs:
        schedule.clear(
            'scheduled_job:{}'.format(scheduled_job_name))
        g_scheduled_jobs.pop(scheduled_job_name, None)
        return


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
    Loud ML job
    """
    func = None
    job_type = None

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.state = 'idle'
        self._result = None
        self.error = None
        self.progress = None
        self.created_dt = datetime.now(pytz.utc)
        self.done_dt = None
        self._future = None
        self.model_name = None

    @property
    def desc(self):
        done_ratio = None
        desc = {
            'id': self.id,
            'type': self.job_type,
            'state': self.state,
        }
        if self.model_name:
            desc['model'] = self.model_name
        if self.result:
            desc['result'] = self._result
        if self.error:
            desc['error'] = self.error
        if self.progress:
            desc['progress'] = self.progress
            done_ratio = float(
                self.progress['eval']) / float(self.progress['max_evals'])
            if done_ratio > 1.0:
                done_ratio = 1.0
        if self.created_dt:
            dt = self.done_dt or datetime.now(pytz.utc)
            duration = (dt - self.created_dt)
            desc['duration'] = duration.total_seconds()
            desc['start_date'] = self.created_dt.strftime('%c')
            desc['start_timestamp'] = self.created_dt.timestamp()
            if done_ratio:
                desc['remaining_time'] = duration.total_seconds() * (
                    1.0 - done_ratio)
        if self.done_dt:
            desc['end_date'] = self.done_dt.strftime('%c')
            desc['end_timestamp'] = self.done_dt.timestamp()
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

    def start(self, config):
        """
        Submit job to worker pool
        """
        global g_pool
        global g_jobs

        self.state = 'waiting'
        self._future = g_pool.schedule(
            loudml.worker.run,
            args=[self.id, 0, self.func, config] + self.args,
            kwargs=self.kwargs,
        )
        self._future.add_done_callback(self._done_cb)
        g_jobs[self.id] = self

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

    def set_final_state(self, state):
        self.done_dt = datetime.now(pytz.utc)
        self.state = state

    def _done_cb(self, result):
        """
        Callback executed when job is done
        """

        try:
            self._result = self._future.result()
            self.set_final_state('done')
            logging.info(
                "job[%s] done. Took %s",
                self.id,
                str(self.done_dt - self.created_dt))
        except concurrent.futures.CancelledError:
            self.error = "job canceled"
            self.set_final_state('canceled')
            logging.error("job[%s] canceled", self.id)
        except Exception as exn:
            self.error = str(exn)
            traceback.print_exc()
            self.set_final_state('failed')
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
        schedule.run_pending()
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


def get_bool_form(param, default=False):
    """
    Read boolean URL parameter
    """
    try:
        return make_bool(request.form.get(param, default))
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


def get_int_form(param, default=None):
    """
    Read integer FORM parameter
    """
    try:
        return int(request.form[param])
    except KeyError:
        return default
    except ValueError:
        raise errors.Invalid("invalid value for parameter '{}'".format(param))


def get_date_form(param, default=None, is_mandatory=False):
    """
    Read date FORM parameter
    """
    try:
        value = request.form[param]
    except KeyError:
        if is_mandatory:
            raise errors.Invalid("'{}' parameter is required".format(param))
        return default

    return schemas.validate(schemas.Timestamp(), value, name=param)


def get_json(is_mandatory=True):
    """
    Parse JSON data from request body
    """
    data = request.json
    if is_mandatory and data is None:
        raise errors.Invalid("request body is empty")

    return data


def get_model_info(name, fields, include_fields):
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

    if fields:
        clear_fields(info, fields, include_fields)
    return info


def get_model_version_info(model_name, model_version, fields, include_fields):
    global g_storage

    model = g_storage.load_model(model_name, ckpt_name=model_version)
    info = model.preview
    info['version'] = {
        'name': model_version,
    }
    if fields:
        clear_fields(info, fields, include_fields)
    return info


def get_template_info(name, fields, include_fields):
    global g_storage

    info = g_storage.get_template_data(name)
    info['params'] = list(g_storage.find_undeclared_variables(name))
    if fields:
        clear_fields(info, fields, include_fields)
    return info


class TemplatesResource(Resource):
    @catch_loudml_error
    def get(self):
        page = get_int_arg('page', default=0)
        per_page = get_int_arg('per_page', default=50)
        if per_page > 100 or per_page <= 0:
            raise errors.Invalid(
                "invalid value for parameter '{}'".format('per_page')
            )
        if page < 0:
            raise errors.Invalid(
                "invalid value for parameter '{}'".format('page')
            )

        include_fields = get_bool_arg('include_fields', default=False)
        if request.args.get('fields'):
            fields = request.args.get('fields').split(";")
        else:
            fields = None

        list_sort_field, list_sort_order = request.args.get(
            'sort', 'name:1').split(':')

        templates = []
        for name in g_storage.list_templates():
            templates.append(get_template_info(
                name, fields, include_fields))

        if (not fields
                or ('settings' in fields and include_fields)
                or ('settings' not in fields and not include_fields)):
            templates = sorted(
                templates,
                key=lambda k: k['settings'].get(list_sort_field),
                reverse=bool(int(list_sort_order) == -1),
            )
        return jsonify(
            templates[page*per_page:(page+1)*per_page])

    @catch_loudml_error
    def post(self):
        global g_storage
        tmpl_name = request.args.get('name')
        if not tmpl_name:
            raise errors.Invalid(
                "invalid value for parameter '{}'".format('name')
            )

        template = loudml.model.load_template(
            settings=request.json,
            name=tmpl_name,
        )

        g_storage.create_template(template)
        return "success", 201


class TemplateResource(Resource):
    @catch_loudml_error
    def get(self, template_names):
        include_fields = get_bool_arg('include_fields', default=False)
        if request.args.get('fields'):
            fields = request.args.get('fields').split(";")
        else:
            fields = None

        return jsonify([
            get_template_info(template_name, fields, include_fields)
            for template_name in template_names.split(';')
        ])

    @catch_loudml_error
    def delete(self, template_names):
        global g_storage

        for template_name in template_names.split(';'):
            g_storage.delete_template(template_name)
            logging.info("template '%s' deleted", template_name)
        return ('', 204)


api.add_resource(TemplatesResource, "/templates")
api.add_resource(TemplateResource, "/templates/<template_names>")


class ModelsResource(Resource):
    @catch_loudml_error
    def get(self):
        page = get_int_arg('page', default=0)
        per_page = get_int_arg('per_page', default=50)
        if per_page > 100 or per_page <= 0:
            raise errors.Invalid(
                "invalid value for parameter '{}'".format('per_page')
            )
        if page < 0:
            raise errors.Invalid(
                "invalid value for parameter '{}'".format('page')
            )

        include_fields = get_bool_arg('include_fields', default=False)
        if request.args.get('fields'):
            fields = request.args.get('fields').split(";")
        else:
            fields = None

        list_sort_field, list_sort_order = request.args.get(
            'sort', 'name:1').split(':')

        models = []
        for name in g_storage.list_models():
            try:
                model = get_model_info(name, fields, include_fields)
                models.append(model)
            except errors.UnsupportedModel:
                continue

        models = sorted(
            models,
            key=lambda k: k['settings'].get(list_sort_field),
            reverse=bool(int(list_sort_order) == -1),
        )
        return jsonify(
            models[page*per_page:(page+1)*per_page])

    @catch_loudml_error
    def post(self):
        global g_storage

        tmpl = request.args.get('from_template', None)
        if tmpl is not None:
            _vars = request.get_json()
            model = g_storage.load_model_from_template(tmpl, **_vars)
        else:
            model = loudml.model.load_model(
                settings=request.json,
            )

        g_storage.create_model(model)

        return "success", 201


class ModelResource(Resource):
    @catch_loudml_error
    def get(self, model_names):
        include_fields = get_bool_arg('include_fields', default=False)
        if request.args.get('fields'):
            fields = request.args.get('fields').split(";")
        else:
            fields = None

        return jsonify([
            get_model_info(model_name, fields, include_fields)
            for model_name in model_names.split(';')
        ])

    @catch_loudml_error
    def delete(self, model_names):
        global g_storage
        global g_training

        for model_name in model_names.split(';'):
            del_scheduled_job(
                '_run({})'.format(model_name))

            job = g_training.get(model_name)
            if job and not job.is_stopped():
                job.cancel()

            g_storage.delete_model(model_name)

            logging.info("model '%s' deleted", model_name)
        return ('', 204)

    @catch_loudml_error
    def head(self, model_names):
        global g_storage
        names = set(model_names.split(';'))
        saved_models = set(g_storage.list_models())
        if len(names & saved_models) == len(names):
            return Response(
                status=200,
            )
        else:
            return Response(
                status=404,
            )

    @catch_loudml_error
    def patch(self, model_names):
        global g_config
        global g_storage

        settings = get_json()

        for model_name in model_names.split(';'):
            settings['name'] = model_name
            model = loudml.model.load_model(settings=settings)

            changes = g_storage.save_model(model, save_state=False)
            for change, param, desc in changes:
                logging.info(
                    "model '%s' %s %s %s",
                    model_name,
                    change,
                    param,
                    desc
                )
                if change == 'change' and param == 'interval':
                    previous_val, next_val = desc
                    scheduled_job_name = '_run({})'.format(model_name)
                    if not scheduled_job_exists(scheduled_job_name):
                        continue
                    del_scheduled_job(scheduled_job_name)
                    new_interval = parse_timedelta(
                        next_val).total_seconds()
                    request_url = '/models/{}/_predict'.format(model_name)
                    add_new_scheduled_job({
                        'name': scheduled_job_name,
                        'method': 'post',
                        'request_url': request_url,
                        'every': {
                            'count': new_interval,
                            'unit': 'seconds',
                        },
                    })

            logging.info("model '%s' updated", model_name)
        return ('', 204)


class ModelVersionsResource(Resource):
    @catch_loudml_error
    def get(self, model_name):
        global g_storage

        page = get_int_arg('page', default=0)
        per_page = get_int_arg('per_page', default=50)
        if per_page > 100 or per_page <= 0:
            raise errors.Invalid(
                "invalid value for parameter '{}'".format('per_page')
            )
        if page < 0:
            raise errors.Invalid(
                "invalid value for parameter '{}'".format('page')
            )

        include_fields = get_bool_arg('include_fields', default=False)
        if request.args.get('fields'):
            fields = request.args.get('fields').split(";")
        else:
            fields = None

        if (fields and
                'version' in fields and not include_fields):
            raise errors.Invalid(
                "'version' cannot be excluded"
            )

        list_sort_field, list_sort_order = request.args.get(
            'sort', 'name:1').split(':')

        g_storage.load_model(model_name)  # raises errors.ModelNotFound()

        models = []
        cur_version = g_storage.get_current_ckpt(model_name)
        for version in g_storage.list_checkpoints(model_name):
            try:
                model = get_model_version_info(
                    model_name, version, fields, include_fields)
                model['version']['active'] = version == cur_version
                models.append(model)
            except errors.UnsupportedModel:
                continue

        models = sorted(
            models,
            key=lambda k: k['version'].get(list_sort_field),
            reverse=bool(int(list_sort_order) == -1),
        )
        return jsonify(
            models[page*per_page:(page+1)*per_page])


api.add_resource(ModelsResource, "/models")
api.add_resource(ModelResource, "/models/<model_names>")
api.add_resource(ModelVersionsResource, "/models/<model_name>/versions")


@app.route("/models/<model_name>/_restore", methods=['POST'])
def model_restore_version(model_name):
    global g_storage
    version = request.args.get('version')
    if not version:
        raise errors.Invalid(
            "invalid value for parameter '{}'".format('version')
        )

    if version not in g_storage.list_checkpoints(model_name):
        raise errors.ModelNotFound(
            name=model_name, version=version)

    g_storage.set_current_ckpt(model_name, version)
    return ('', 204)


@app.route("/models/<model_name>/_train", methods=['POST'])
def model_train(model_name):
    global g_storage
    global g_training
    global g_config

    g_storage.load_model(model_name)
    kwargs = {}

    kwargs['from_date'] = get_date_arg('from', is_mandatory=True)
    kwargs['to_date'] = get_date_arg('to', default="now")
    if get_bool_arg('autostart'):
        kwargs.update({
            'autostart': True,
            'save_prediction': get_bool_arg('save_prediction'),
            'output_bucket': request.args.get('output_bucket'),
            'detect_anomalies': get_bool_arg('detect_anomalies'),
        })

    bucket = request.args.get('input')
    if bucket is not None:
        kwargs['bucket'] = bucket

    max_evals = get_int_arg('max_evals')
    if max_evals is not None:
        kwargs['max_evals'] = max_evals

    epochs = get_int_arg('epochs')
    if epochs is not None:
        kwargs['num_epochs'] = epochs

    job = TrainingJob(model_name, **kwargs)
    job.start(g_config)

    g_training[model_name] = job

    return jsonify(job.id), 202


class HooksResource(Resource):
    @catch_loudml_error
    def get(self, model_name):
        global g_storage

        return jsonify(g_storage.list_model_hooks(model_name))

    @catch_loudml_error
    def post(self, model_name):
        global g_storage

        data = get_json()

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

        data = get_json()

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


def _remove_bucket_secrets(bucket):
    bucket.pop('password', None)
    bucket.pop('dbuser_password', None)
    bucket.pop('write_token', None)
    bucket.pop('read_token', None)


class BucketsResource(Resource):
    @catch_loudml_error
    def get(self):
        global g_config
        page = get_int_arg('page', default=0)
        per_page = get_int_arg('per_page', default=50)
        if per_page > 100 or per_page <= 0:
            raise errors.Invalid(
                "invalid value for parameter '{}'".format('per_page')
            )
        if page < 0:
            raise errors.Invalid(
                "invalid value for parameter '{}'".format('page')
            )

        include_fields = get_bool_arg('include_fields', default=False)
        if request.args.get('fields'):
            fields = request.args.get('fields').split(";")
        else:
            fields = None

        list_sort_field, list_sort_order = request.args.get(
            'sort', 'name:1').split(':')

        buckets = []
        for bucket in g_config.buckets.values():
            _remove_bucket_secrets(bucket)
            if fields:
                clear_fields(bucket, fields, include_fields)
            buckets.append(bucket)

        buckets = sorted(
            buckets,
            key=lambda k: k.get(list_sort_field),
            reverse=bool(int(list_sort_order) == -1),
        )
        return jsonify(
            buckets[page*per_page:(page+1)*per_page])

    @catch_loudml_error
    def post(self):
        global g_config

        new = request.get_json()
        g_config.put_bucket(new)
        return ('', 201)


class BucketResource(Resource):
    @catch_loudml_error
    def get(self, bucket_names):
        global g_config
        include_fields = get_bool_arg('include_fields', default=False)
        if request.args.get('fields'):
            fields = request.args.get('fields').split(";")
        else:
            fields = None

        buckets = []
        for bucket_name in bucket_names.split(';'):
            bucket = g_config.get_bucket(bucket_name)
            _remove_bucket_secrets(bucket)
            if fields:
                clear_fields(bucket, fields, include_fields)
            buckets.append(bucket)

        return jsonify(buckets)

    @catch_loudml_error
    def patch(self, bucket_names):
        global g_config

        data = request.get_json()
        for bucket_name in bucket_names.split(';'):
            data['name'] = bucket_name
            g_config.patch_bucket(data)
            logging.info("bucket '%s' changed", bucket_name)
        return ('', 204)

    @catch_loudml_error
    def delete(self, bucket_names):
        global g_config
        for bucket_name in bucket_names.split(';'):
            g_config.del_bucket(bucket_name)
            logging.info("bucket '%s' deleted", bucket_name)
        return ('', 204)

    @catch_loudml_error
    def head(self, bucket_names):
        global g_config
        names = set(bucket_names.split(';'))
        saved_buckets = set(g_config.list_buckets())
        if len(names & saved_buckets) == len(names):
            return Response(
                status=200,
            )
        else:
            return Response(
                status=404,
            )


api.add_resource(BucketsResource, "/buckets")
api.add_resource(BucketResource, "/buckets/<bucket_names>")


@app.route("/buckets/<bucket_name>/_clear", methods=['POST'])
def bucket_clear(bucket_name):
    global g_config
    settings = g_config.get_bucket(bucket_name)
    bucket = load_bucket(settings)
    bucket.drop()
    return ('', 204)


@app.route("/buckets/<bucket_name>/_write", methods=['POST'])
def bucket_write(bucket_name):
    global g_config
    _ = g_config.get_bucket(bucket_name)

    points = get_json()
    job = WriteBucketJob(
        bucket_name=bucket_name,
        points=points,
        **request.args
    )
    job.start(g_config)
    return str(job.id), 202


@app.route("/buckets/<bucket_name>/_read", methods=['POST'])
def bucket_read(bucket_name):
    global g_config
    _ = g_config.get_bucket(bucket_name)

    from_date = get_date_arg('from', is_mandatory=True)
    to_date = get_date_arg('to', is_mandatory=True)
    bucket_interval = parse_timedelta(
        request.args.get('bucket_interval', 0)).total_seconds()
    if not bucket_interval:
        raise errors.Invalid(
            "invalid value for parameter 'bucket_interval'")

    features = []
    for feature in request.args.get('features', '').split(';'):
        field = None
        metric = None
        measurement = None
        for index, val in parse_expression("({})".format(feature)):
            if index > 1:
                raise errors.Invalid(
                    "invalid value for parameter 'features'")
            if index == 0:
                metric = val.split('(')[0]
            elif index == 1:
                if '.' in val:
                    measurement, field = val.split('.')
                else:
                    field = val
        if not field or not metric:
            raise errors.Invalid(
                "invalid value for parameter 'features'")
        features.append(
            loudml.model.Feature(
                name='{}_{}'.format(metric, field),
                measurement=measurement,
                field=field,
                metric=metric,
            )
        )

    job = ReadBucketJob(
        bucket_name=bucket_name,
        from_date=from_date,
        to_date=to_date,
        bucket_interval=bucket_interval,
        features=features,
    )
    job.start(g_config)
    return str(job.id), 202


class JobsResource(Resource):
    @catch_loudml_error
    def get(self):
        global g_jobs
        page = get_int_arg('page', default=0)
        per_page = get_int_arg('per_page', default=50)
        if per_page > 100 or per_page <= 0:
            raise errors.Invalid(
                "invalid value for parameter '{}'".format('per_page')
            )
        if page < 0:
            raise errors.Invalid(
                "invalid value for parameter '{}'".format('page')
            )

        include_fields = get_bool_arg('include_fields', default=False)
        if request.args.get('fields'):
            fields = request.args.get('fields').split(";")
        else:
            fields = None

        list_sort_field, list_sort_order = request.args.get(
            'sort', 'id:1').split(':')

        jobs = []
        for entry in g_jobs.values():
            job = entry.desc
            if fields:
                clear_fields(job, fields, include_fields)
            jobs.append(job)

        jobs = sorted(
            jobs,
            key=lambda k: k.get(list_sort_field),
            reverse=bool(int(list_sort_order) == -1),
        )
        return jsonify(
            jobs[page*per_page:(page+1)*per_page])


class JobResource(Resource):
    @catch_loudml_error
    def get(self, job_ids):
        global g_jobs
        include_fields = get_bool_arg('include_fields', default=False)
        if request.args.get('fields'):
            fields = request.args.get('fields').split(";")
        else:
            fields = None

        wanted = set(job_ids.split(';'))

        jobs = [
            get_job_desc(job_id, fields, include_fields)
            for job_id in (wanted & set(g_jobs.keys()))
        ]
        if not len(jobs):
            return "job(s) not found", 404

        return jsonify(jobs)

    @catch_loudml_error
    def head(self, job_ids):
        global g_jobs
        ids = set(job_ids.split(';'))
        found_ids = set(g_jobs.keys())
        if len(ids & found_ids) == len(ids):
            return Response(
                status=200,
            )
        else:
            return Response(
                status=404,
            )


api.add_resource(JobsResource, "/jobs")
api.add_resource(JobResource, "/jobs/<job_ids>")


class ScheduledJobsResource(Resource):
    @catch_loudml_error
    def get(self):
        global g_scheduled_jobs
        page = get_int_arg('page', default=0)
        per_page = get_int_arg('per_page', default=50)
        if per_page > 100 or per_page <= 0:
            raise errors.Invalid(
                "invalid value for parameter '{}'".format('per_page')
            )
        if page < 0:
            raise errors.Invalid(
                "invalid value for parameter '{}'".format('page')
            )

        include_fields = get_bool_arg('include_fields', default=False)
        if request.args.get('fields'):
            fields = request.args.get('fields').split(";")
        else:
            fields = None

        list_sort_field, list_sort_order = request.args.get(
            'sort', 'name:1').split(':')

        jobs = []
        for job in g_scheduled_jobs.values():
            if fields:
                clear_fields(job, fields, include_fields)
            jobs.append(job)

        jobs = sorted(
            jobs,
            key=lambda k: k.get(list_sort_field),
            reverse=bool(int(list_sort_order) == -1),
        )
        return jsonify(
            jobs[page*per_page:(page+1)*per_page])

    @catch_loudml_error
    def post(self):
        desc = schemas.validate(schemas.ScheduledJob, get_json())
        add_new_scheduled_job(desc)
        return ('', 201)

    @catch_loudml_error
    def delete(self):
        global g_scheduled_jobs
        schedule.clear('scheduled_job')
        g_scheduled_jobs.clear()
        return ('', 204)


class ScheduledJobResource(Resource):
    @catch_loudml_error
    def get(self, job_ids):
        global g_scheduled_jobs
        include_fields = get_bool_arg('include_fields', default=False)
        if request.args.get('fields'):
            fields = request.args.get('fields').split(";")
        else:
            fields = None

        wanted = set(job_ids.split(';'))

        jobs = [
            get_sched_job_desc(
                job_id, fields, include_fields)
            for job_id in (wanted & set(g_scheduled_jobs.keys()))
        ]
        if not len(jobs):
            return "job(s) not found", 404

        return jsonify(jobs)

    @catch_loudml_error
    def head(self, job_ids):
        global g_scheduled_jobs
        ids = set(job_ids.split(';'))
        found_ids = set(g_scheduled_jobs.keys())
        if len(ids & found_ids) == len(ids):
            return Response(
                status=200,
            )
        else:
            return Response(
                status=404,
            )

    @catch_loudml_error
    def delete(self, job_ids):
        global g_scheduled_jobs
        for job_id in job_ids.split(';'):
            schedule.clear(
                'scheduled_job:{}'.format(job_id))
            g_scheduled_jobs.pop(job_id, None)
        return ('', 204)


api.add_resource(ScheduledJobsResource, "/scheduled_jobs")
api.add_resource(ScheduledJobResource, "/scheduled_jobs/<job_ids>")


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
            'output_bucket': kwargs.pop('output_bucket', None),
            'detect_anomalies': kwargs.pop('detect_anomalies', False),
            'from_date': kwargs.get('from_date', None),
        }
        self._kwargs = kwargs

    def start(self, config):
        """
        Submit training job to worker pool
        """
        global g_training_pool
        global g_nice
        global g_jobs

        g_jobs[self.id] = self
        self.state = 'waiting'
        self._future = g_training_pool.schedule(
            loudml.worker.run,
            args=[self.id, g_nice, self.func, config] + self.args,
            kwargs=self.kwargs,
        )
        self._future.add_done_callback(self._done_cb)

    def _done_cb(self, result):
        """
        Callback executed when job is done
        """
        super()._done_cb(result)
        if self.state == 'done' and self.autostart:
            logging.info(
                "scheduling autostart for model '%s'",
                self.model_name
            )
            model = g_storage.load_model(self.model_name)
            params = self._kwargs_start.copy()
            params.pop('from_date')
            model.set_run_params(params)
            g_storage.save_model(model)
            try:
                _model_start(model, self._kwargs_start)
            except errors.LoudMLException:
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


class ReadBucketJob(Job):
    """
    Read data from bucket
    """
    func = 'read_from_bucket'
    job_type = 'read'

    def __init__(
        self,
        bucket_name,
        from_date,
        to_date,
        bucket_interval,
        features,
    ):
        super().__init__()
        self.bucket_name = bucket_name
        self.from_date = from_date
        self.to_date = to_date
        self.bucket_interval = bucket_interval
        self.features = features

    @property
    def args(self):
        return [
            self.bucket_name,
            self.from_date,
            self.to_date,
            self.bucket_interval,
            self.features,
        ]


class WriteBucketJob(Job):
    """
    Write data points to bucket
    """
    func = 'write_to_bucket'
    job_type = 'write'

    def __init__(
        self,
        bucket_name,
        points,
        **kwargs
    ):
        super().__init__()
        self.bucket_name = bucket_name
        self.points = points
        self._kwargs = kwargs

    @property
    def args(self):
        return [
            self.bucket_name,
            self.points,
        ]

    @property
    def kwargs(self):
        return self._kwargs


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
    global g_config
    scheduled_job_name = '_run({})'.format(model.name)
    if scheduled_job_exists(scheduled_job_name):
        return  # idempotent _start

    def create_job(from_date=None, save_run_state=True, detect_anomalies=None):
        kwargs = params.copy()
        if detect_anomalies is not None:
            kwargs['detect_anomalies'] = detect_anomalies

        to_date = datetime.now(pytz.utc).timestamp() - model.offset

        if model.type in ['timeseries', 'donut']:
            if from_date is None:
                from_date = to_date - model.interval

            kwargs['from_date'] = from_date
            kwargs['to_date'] = to_date

        job = PredictionJob(
            model.name,
            save_run_state=save_run_state,
            **kwargs
        )

        job.start(g_config)

    from_date = params.pop('from_date', None)
    create_job(from_date, save_run_state=False, detect_anomalies=False)


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
        output_bucket=request.args.get('output_bucket'),
        detect_anomalies=request.args.get('detect_anomalies', default=False),
    )
    job.start(g_config)

    if get_bool_arg('bg', default=False):
        return str(job.id), 202

    return jsonify(job.result())


@app.route("/models/<model_name>/_top")
def model_top(model_name):
    global g_storage
    global g_config

    from_date = get_date_arg('from', is_mandatory=True)
    to_date = get_date_arg('to', is_mandatory=True)
    size = get_int_arg('size', default=10)

    model = g_storage.load_model(model_name)

    src_settings = g_config.get_bucket(model.default_bucket)
    source = load_bucket(src_settings)

    res = source.get_top_abnormal_keys(
        model,
        from_date,
        to_date,
        size,
    )

    return jsonify(res)


@app.route("/models/<model_name>/_start", methods=['POST'])
def model_start(model_name):
    global g_storage

    params = {
        'save_prediction': get_bool_arg('save_prediction'),
        'output_bucket': request.args.get('output_bucket'),
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
    global g_storage

    scheduled_job_name = '_run({})'.format(model_name)
    if not scheduled_job_exists(scheduled_job_name):
        return "model is not active", 404

    del_scheduled_job(scheduled_job_name)
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
    global g_config

    params = {
        'save_prediction': get_bool_arg('save_prediction'),
        'output_bucket': request.args.get('output_bucket'),
    }

    model = g_storage.load_model(model_name)

    params['from_date'] = get_date_arg('from', default='now')
    params['to_date'] = get_date_arg('to', is_mandatory=True)

    constraint = request.args.get('constraint')
    if constraint:
        params['constraint'] = parse_constraint(constraint)

    job = ForecastJob(model.name, **params)
    job.start(g_config)

    if get_bool_arg('bg', default=False):
        return str(job.id), 202

    return jsonify(job.result())

#
# Example of job
#
# class DummyJob(Job):
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
# @app.route("/do-things")
# def do_things():
#    job = DummyJob(int(request.args.get('value', 0)))
#    job.start()
#    return str(job.id), 202


@app.route("/")
def slash():
    version = pkg_resources.get_distribution("loudml").version
    return jsonify({
        'version': version,
        'tagline': "The Disruptive Machine Learning API",
        'host_id': my_host_id(),
    })


@app.errorhandler(403)
def err_forbidden(e):
    return "forbidden", 403


@app.errorhandler(404)
def err_not_found(e):
    return "unknown endpoint", 404


@app.errorhandler(405)
def err_now_allowed(e):
    return "method not allowed", 405


@app.errorhandler(410)
def err_gone(e):
    return "gone", 410


@app.errorhandler(500)
def err_internal(e):
    return "internal server error", 500


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


def g_app_init(path):
    global g_config
    global g_training_pool
    global g_nice
    global g_pool
    global g_queue
    global g_storage
    global g_timer

    g_config = loudml.config.load_config(path)
    g_storage = FileStorage(g_config.storage['path'])
    g_queue = multiprocessing.Queue()
    g_nice = g_config.training.get('nice', 0)
    g_training_pool = pebble.ProcessPool(
        max_workers=g_config.server.get('workers', 1),
        max_tasks=g_config.server.get('maxtasksperchild', 1),
        initializer=loudml.worker.init_worker,
        initargs=[g_queue],
    )
    g_pool = pebble.ProcessPool(
        max_workers=g_config.server.get('workers', 1),
        max_tasks=g_config.server.get('maxtasksperchild', 1),
        initializer=loudml.worker.init_worker,
        initargs=[g_queue],
    )
    g_timer = RepeatingTimer(1, read_messages)
    g_timer.start()

    def daemon_send_metrics():
        send_metrics(g_config.metrics, g_storage, user_agent="loudmld")

    daemon_send_metrics()
    schedule.every().hour.do(daemon_send_metrics)

    def daemon_clear_jobs():
        global g_jobs
        duration = g_config.server.get('jobs_max_ttl')
        now_dt = datetime.now(pytz.utc)
        expired = [
            job.id
            for job in g_jobs.values()
            if (job.is_stopped() and
                (now_dt - job.done_dt) > timedelta(seconds=duration))
        ]
        for i in expired:
            del g_jobs[i]

    schedule.every().minute.do(daemon_clear_jobs)


def g_app_stop():
    global g_timer
    global g_pool
    global g_training_pool
    global g_config
    global g_nice
    global g_queue

    schedule.clear('bg')
    g_timer.cancel()
    g_pool.stop()
    g_pool.join()
    g_training_pool.stop()
    g_training_pool.join()
    g_config = None
    g_nice = 0
    g_pool = None
    g_queue = None
    g_timer = None


def main():
    """
    Loud ML server
    """

    global g_config

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

    try:
        g_app_init(args.config)
        loudml.config.load_plugins(args.config)
    except errors.LoudMLException as exn:
        logging.error(exn)
        sys.exit(1)

    try:
        cron = CronTab(user='loudml')
        cron.remove_all()
        if g_config.training['incremental']['enable']:
            for tab in g_config.training['incremental']['crons']:
                job = cron.new(command='/usr/bin/loudml train \* -i -f {} -t {}'.format(tab['from'], tab['to']),  # noqa W605
                               comment='incremental training')
                job.setall(tab['crontab'])

        for item in cron:
            logging.info(item)

        cron.write()
    except OSError:
        logging.error(
            "detected development environment - incremental training disabled"
        )

    listen_addr = g_config.server['listen']
    host, port = listen_addr.split(':')

    restart_predict_jobs()

    try:
        http_server = WSGIServer((host, int(port)), app)
        logging.info("starting Loud ML server on %s", listen_addr)
        http_server.serve_forever()
    except OSError as exn:
        logging.error(str(exn))
    except KeyboardInterrupt:
        pass

    logging.info("stopping")
    g_app_stop()

"""
LoudML server
"""

import argparse
import logging
import multiprocessing
import multiprocessing.pool
import queue
import sys
import uuid

import loudml.config
import loudml.model
import loudml.worker

from threading import Timer

from flask import (
    Flask,
    jsonify,
    request,
)
from flask_restful import (
    Api,
    Resource,
)
from . import (
    errors,
)
from .filestorage import (
    FileStorage,
)

app = Flask(__name__, static_url_path='/static', template_folder='templates')
api = Api(app)

g_config = None
g_jobs = {}
g_training = {}
g_storage = None
g_pool = None
g_queue = None

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
        logging.info("job done")

    def fail(self, error):
        """
        Callback executed when job fails
        """
        self.state = 'failed'
        self.error = str(error)
        logging.info("job failed: %s", self.error)


def set_job_state(job_id, state):
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

def read_messages():
    """
    Read messages from subprocesses
    """
    global g_queue

    while True:
        try:
            msg = g_queue.get(block=False)

            if msg['type'] == 'job_state':
                set_job_state(msg['job_id'], msg['state'])
        except queue.Empty:
            break


class ModelsResource(Resource):
    def get(self):
        global g_storage

        models = []

        for name in g_storage.list_models():
            model = g_storage.load_model(name)
            models.append({
                'settings': model.settings,
            })

        return jsonify(models)

    def put(self):
        global g_storage

        model = loudml.model.load_model(settings=request.json)

        try:
            g_storage.create_model(model)
        except errors.ModelExists as exn:
            return str(exn), 409

        return "success", 201


class ModelResource(Resource):
    def get(self, model_name):
        global g_storage

        try:
            model = g_storage.load_model(model_name)
        except errors.ModelNotFound as exn:
            return str(exn), 404

        return jsonify(model.settings)

    def delete(self, model_name):
        global g_storage

        try:
            g_storage.delete_model(model_name)
        except errors.ModelNotFound as exn:
            return str(exn), 404

        return "success"

    def post(self, model_name):
        global g_storage

        settings = request.json
        settings['name'] = model_name
        model = loudml.model.load_model(settings=request.json)

        try:
            g_storage.delete_model(model_name)
        except errors.ModelNotFound:
            pass

        g_storage.create_model(model)
        return "success"


api.add_resource(ModelsResource, "/models")
api.add_resource(ModelResource, "/models/<model_name>")

class DataSourcesResource(Resource):
    def get(self):
        return jsonify(list(g_config.datasources.values()))


class DataSourceResource(Resource):
    def get(self, datasource_name):
        try:
            datasource = g_config.get_datasource(datasource_name)
        except errors.DataSourceNotFound as exn:
            return str(exn), 404

        return jsonify(datasource)


api.add_resource(DataSourcesResource, "/datasources")
api.add_resource(DataSourceResource, "/datasources/<datasource_name>")

class JobsResource(Resource):
    def get(self):
        global g_jobs
        return jsonify([job.desc for job in g_jobs.values()])


class JobResource(Resource):
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


class TrainingJobsResource(Resource):
    def get(self, model_name):
        global g_training

        jobs = g_training.get(model_name, {})
        return jsonify([job.desc for job in jobs.values()])

    def put(self, model_name):
        global g_storage
        global g_training

        try:
            model = g_storage.load_model(model_name)
        except errors.ModelNotFound as exn:
            return str(exn), 404

        kwargs = {}

        if model.type == 'timeseries':
            kwargs['from_date'] = request.args.get('from', "now-1d")
            kwargs['to_date'] = request.args.get('to', "now")

        job = TrainingJob(model_name, **kwargs)
        job.start()

        if model_name not in g_training:
            g_training[model_name] = {}

        g_training[model_name][job.id] = job

        return jsonify(job.id)


class TrainingJobResource(Resource):
    def get(self, model_name, job_id):
        global g_jobs

        jobs = g_training.get(model_name, {})
        job = jobs.get(job_id)

        if job is None:
            return "training job not found", 404

        return jsonify(job.desc)

api.add_resource(TrainingJobsResource, "/training/<model_name>")
api.add_resource(TrainingJobResource, "/training/<model_name>/<job_id>")

"""
# Example of job
#
class DummyJob(Job):
    func = 'do_things'
    job_type = 'dummy'

    def __init__(self, value):
        super().__init__()
        self.value = value

    @property
    def args(self):
        return [self.value]

# Example of endpoint that submits jobs
@app.route("/do-things")
def do_things():
    job = DummyJob(int(request.args.get('value', 0)))
    job.start()
    return str(job.id)
"""

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

    try:
        g_config = loudml.config.load_config(args.config)
        g_storage = FileStorage(g_config.storage['path'])
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

    try:
        app.run(host=host, port=int(port))
    except KeyboardInterrupt:
        pass

    logging.info("stopping")
    timer.cancel()
    g_pool.close()
    g_pool.join()

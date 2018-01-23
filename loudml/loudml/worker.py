"""
LoudML worker
"""

import logging
import signal

import loudml.config
import loudml.datasource
import loudml.model

from loudml import (
    errors,
)

from loudml.filestorage import (
    FileStorage,
)

g_worker = None

class Worker:
    """
    LoudML worker
    """

    def __init__(self, config_path, msg_queue):
        self.config = loudml.config.load_config(config_path)
        self.storage = FileStorage(self.config.storage['path'])
        self._msg_queue = msg_queue
        self.job_id = None
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def run(self, job_id, func_name, *args, **kwargs):
        """
        Run requested task and return the result
        """

        self._msg_queue.put({
            'type': 'job_state',
            'job_id': job_id,
            'state': 'running',
        })
        logging.info("job[%s] starting", job_id)
        self.job_id = job_id

        try:
            res = getattr(self, func_name)(*args, **kwargs)
        except errors.LoudMLException as exn:
            raise exn
        except Exception as exn:
            logging.exception(exn)
            raise exn
        finally:
            self.job_id = None

        return res

    def train(self, model_name, **kwargs):
        """
        Train model
        """

        model = self.storage.load_model(model_name)
        src_settings = self.config.get_datasource(model.default_datasource)
        source = loudml.datasource.load_datasource(src_settings)
        model.train(source, **kwargs)
        self.storage.save_model(model)

        # TODO return loss and accuracy

    def predict(self, model_name, **kwargs):
        """
        Ask model for a prediction
        """

        model = self.storage.load_model(model_name)
        src_settings = self.config.get_datasource(model.default_datasource)
        source = loudml.datasource.load_datasource(src_settings)
        prediction = model.predict(source, **kwargs)

        if model.type == 'timeseries':
            logging.info("job[%s] predicted values for %d time buckets",
                         self.job_id, len(prediction.timestamps))
            source.save_timeseries_prediction(prediction, model)
        else:
            logging.info("job[%s] prediction done", self.job_id)

        return prediction.format_series()


    """
    # Example
    #
    def do_things(self, value):
        if value:
        import time
        time.sleep(value)
        return {'value': value}
    else:
        raise Exception("no value")
    """


def init_worker(config_path, msg_queue):
    global g_worker
    g_worker = Worker(config_path, msg_queue)

def run(job_id, func_name, *args, **kwargs):
    global g_worker
    return g_worker.run(job_id, func_name, *args, **kwargs)

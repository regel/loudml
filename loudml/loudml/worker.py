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

    def train(self, model_name, datasource=None, **kwargs):
        """
        Train model
        """

        model = self.storage.load_model(model_name)

        src_name = datasource or model.default_datasource
        src_settings = self.config.get_datasource(src_name)
        source = loudml.datasource.load_datasource(src_settings)

        def progress_cb(current_eval, max_evals):
            self._msg_queue.put({
                'type': 'job_state',
                'job_id': self.job_id,
                'state': 'running',
                'progress': {
                    'eval': current_eval,
                    'max_evals': max_evals,
                },
            })

        model.train(
            source,
            progress_cb=progress_cb,
            **kwargs
        )
        self.storage.save_model(model)

    def predict(
        self,
        model_name,
        save_prediction=False,
        save_scores=False,
        detect_anomalies=False,
        **kwargs
    ):
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
            if save_prediction:
                source.save_timeseries_prediction(prediction, model)
            if detect_anomalies:
                hooks = self.storage.load_model_hooks(model_name)
                model.detect_anomalies(prediction, hooks)
                self.storage.save_model(model)

                if save_scores:
                    for bucket in prediction.format_buckets():
                        stats = bucket.get('stats')
                        score = stats.get('score')
                        is_anomaly = stats.get('anomaly')
                        source.insert_times_data(
                            ts=bucket['timestamp'],
                            data={ 'score': score },
                            tags={ 'anomaly': is_anomaly },
                            measurement='scores_{}'.format(model.name),
                        )

                # TODO .detect_anomalies() produces warning messages
                # and store anomalies into 'prediction'.
                # Now, we can get them using 'prediction.get_anomalies()'
                # and store them anywhere

            fmt = kwargs.get('format', 'series')

            if fmt == 'buckets':
                return prediction.format_buckets()
            elif fmt == 'series':
                return prediction.format_series()
            else:
                raise errors.Invalid('unknown requested format')

        elif model.type == 'fingerprints':
            logging.info("job[%s]: computing fingerprints for model '%s'",
                         self.job_id, self.model.name)
            if save_prediction:
                self.storage.save_model(model)
        else:
            logging.info("job[%s] prediction done", self.job_id)

    def forecast(
        self,
        model_name,
        save_prediction=False,
        **kwargs
    ):
        """
        Ask model for a forecast
        """

        model = self.storage.load_model(model_name)
        src_settings = self.config.get_datasource(model.default_datasource)
        source = loudml.datasource.load_datasource(src_settings)

        constraint = kwargs.pop('constraint')

        forecast = model.forecast(source, **kwargs)

        if model.type == 'timeseries':
            logging.info("job[%s] forecasted values for %d time buckets",
                         self.job_id, len(forecast.timestamps))
            if constraint:
                model.test_constraint(
                    forecast,
                    constraint['feature'],
                    constraint['type'],
                    constraint['threshold'],
                )

            if save_prediction:
                source.save_timeseries_prediction(forecast, model)

            return forecast.format_series()
        else:
            logging.info("job[%s] forecast done", self.job_id)


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

"""
Loud ML worker
"""

import logging
import signal
import os

import loudml.config
import loudml.bucket
import loudml.model
from loudml.misc import make_ts
from loudml import (
    errors,
)

from loudml.filestorage import (
    FileStorage,
)

g_worker = None


class Worker:
    """
    Loud ML worker
    """

    def __init__(self, msg_queue):
        self.storage = None
        self._msg_queue = msg_queue
        self.job_id = None
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def run(self, job_id, nice, func_name, config, *args, **kwargs):
        """
        Run requested task and return the result
        """

        self._msg_queue.put({
            'type': 'job_state',
            'job_id': job_id,
            'state': 'running',
        })
        logging.info("job[%s] starting, nice=%d", job_id, nice)
        self.job_id = job_id
        self.config = config
        self.storage = FileStorage(config.storage['path'])
        curnice = os.nice(0)
        os.nice(int(nice) - curnice)

        try:
            res = getattr(self, func_name)(*args, **kwargs)
        except errors.LoudMLException as exn:
            raise exn
        except Exception as exn:
            logging.exception(exn)
            raise exn
        finally:
            self.job_id = None
            self.config = None
            self.storage = None

        return res

    def train(self, model_name, bucket=None, **kwargs):
        """
        Train model
        """

        model = self.storage.load_model(model_name)

        bucket_name = bucket or model.default_bucket
        bucket_settings = self.config.get_bucket(bucket_name)
        bucket = loudml.bucket.load_bucket(bucket_settings)

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
        windows = bucket.list_anomalies(
            kwargs['from_date'],
            kwargs['to_date'],
            tags={'model': model_name},
        )
        num_epochs = kwargs.pop('num_epochs', self.config.training['epochs'])
        model.train(
            bucket,
            batch_size=self.config.training['batch_size'],
            num_epochs=num_epochs,
            num_cpus=self.config.training['num_cpus'],
            num_gpus=self.config.training['num_gpus'],
            progress_cb=progress_cb,
            windows=windows,
            **kwargs
        )
        self.storage.save_model(model)

    def _save_timeseries_prediction(
        self,
        model,
        prediction,
        input_bucket,
        output_bucket=None,
    ):
        if output_bucket is None:
            output_bucket = model.default_bucket

        if output_bucket is None or output_bucket == input_bucket.name:
            bucket = input_bucket
        else:
            try:
                bucket_settings = self.config.get_bucket(
                    output_bucket
                )
                bucket = loudml.bucket.load_bucket(bucket_settings)
            except errors.LoudMLException as exn:
                logging.error("cannot load bucket: %s", str(exn))
                return

        bucket.init(data_schema=prediction.get_schema())
        bucket.save_timeseries_prediction(prediction, tags=model.get_tags())

    def read_from_bucket(
        self,
        bucket_name,
        from_date,
        to_date,
        bucket_interval,
        features,
    ):
        """
        Run query in the bucket TSDB and return data
        """
        bucket_settings = self.config.get_bucket(bucket_name)
        bucket = loudml.bucket.load_bucket(bucket_settings)

        data = bucket.get_times_data(
            bucket_interval=bucket_interval,
            features=features,
            from_date=from_date,
            to_date=to_date,
        )
        timestamps = []
        obs = {
            feature.name: []
            for feature in features
        }
        for (_, values, timeval) in data:
            timestamps.append(make_ts(timeval))
            for (feature, val) in zip(features, values):
                obs[feature.name].append(float(val))

        return {
            'timestamps': timestamps,
            'observed': obs,
        }

    def write_to_bucket(
        self,
        bucket_name,
        points,
        **kwargs
    ):
        """
        Writes data points to the bucket TSDB
        """
        bucket_settings = self.config.get_bucket(bucket_name)
        bucket = loudml.bucket.load_bucket(bucket_settings)

        fields = [
            list(point.keys())
            for point in points
        ]
        flat_fields = [
            field for sub_fields in fields for field in sub_fields
        ]
        fields = set(flat_fields) - set(['timestamp', 'tags'])

        tags = [
            list(point['tags'].keys())
            for point in points
            if 'tags' in point
        ]
        flat_tags = [
            tag for sub_tags in tags for tag in sub_tags
        ]
        tags = set(flat_tags)

        data_schema = {}
        data_schema.update({
            tag: {"type": "keyword"}
            for tag in tags
        })
        data_schema.update({
            field: {"type": "float"}
            for field in fields
        })
        bucket.init(data_schema=data_schema)
        for point in points:
            ts = make_ts(point.pop('timestamp'))
            tags = point.pop('tags', None)
            bucket.insert_times_data(
                ts=ts,
                data=point,
                tags=tags,
                **kwargs
            )

        bucket.commit()

    def predict(
        self,
        model_name,
        save_run_state=True,
        save_prediction=False,
        detect_anomalies=False,
        output_bucket=None,
        **kwargs
    ):
        """
        Ask model for a prediction
        """

        model = self.storage.load_model(model_name)
        bucket_settings = self.config.get_bucket(model.default_bucket)
        bucket = loudml.bucket.load_bucket(bucket_settings)

        if model.type in ['timeseries', 'donut']:
            _state = model.get_run_state()
            if detect_anomalies:
                prediction = model.predict2(
                    bucket,
                    _state=_state,
                    num_cpus=self.config.inference['num_cpus'],
                    num_gpus=self.config.inference['num_gpus'],
                    **kwargs
                )
            else:
                prediction = model.predict(
                    bucket,
                    num_cpus=self.config.inference['num_cpus'],
                    num_gpus=self.config.inference['num_gpus'],
                    **kwargs
                )

            logging.info("job[%s] predicted values for %d time buckets",
                         self.job_id, len(prediction.timestamps))
            if detect_anomalies:
                hooks = self.storage.load_model_hooks(
                    model.settings,
                    bucket,
                )
                model.detect_anomalies(prediction, hooks)
            if save_run_state:
                model.set_run_state(_state)
                self.storage.save_state(model)
            if save_prediction:
                self._save_timeseries_prediction(
                    model,
                    prediction,
                    bucket,
                    output_bucket,
                )

            fmt = kwargs.get('format', 'series')

            if fmt == 'buckets':
                return prediction.format_buckets()
            elif fmt == 'series':
                return prediction.format_series()
            else:
                raise errors.Invalid('unknown requested format')

        else:
            logging.info("job[%s] prediction done", self.job_id)

    def forecast(
        self,
        model_name,
        save_prediction=False,
        output_bucket=None,
        **kwargs
    ):
        """
        Ask model for a forecast
        """

        model = self.storage.load_model(model_name)
        bucket_settings = self.config.get_bucket(model.default_bucket)
        bucket = loudml.bucket.load_bucket(bucket_settings)

        constraint = kwargs.pop('constraint', None)

        forecast = model.forecast(
            bucket,
            num_cpus=self.config.inference['num_cpus'],
            num_gpus=self.config.inference['num_gpus'],
            **kwargs
        )

        if model.type in ['timeseries', 'donut']:
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
                self._save_timeseries_prediction(
                    model,
                    forecast,
                    bucket,
                    output_bucket,
                )

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


def init_worker(msg_queue):
    global g_worker
    g_worker = Worker(msg_queue)


def run(job_id, nice, func_name, *args, **kwargs):
    global g_worker
    return g_worker.run(job_id, nice, func_name, *args, **kwargs)

"""
Base interface for Loud ML bucket classes
"""
import datetime

from abc import (
    ABCMeta,
    abstractmethod,
)

from voluptuous import (
    ALLOW_EXTRA,
    All,
    Length,
    Optional,
    Range,
    Required,
    Schema,
    Any,
)

from loudml import (
    errors,
    misc,
    schemas,
)


class Bucket(metaclass=ABCMeta):
    """
    Loud ML abstract Bucket class. Provides a standard interface
    that can be inherited in order to aggregate, read, and write
    data points to a TSDB.
    This class hides TSDB vendor specific logic and helps to maintain
    a uniform Loud ML experience across different TSDB vendors.
    """

    SCHEMA = Schema({
        Required('name'): All(schemas.key, Length(max=256)),
        Required('type'): All(schemas.key, Length(max=256)),
        Optional('max_series_per_request', default=2000): All(
            int,
            Range(min=1),
        ),
        'timestamp_field': Any(None, schemas.key),
    }, extra=ALLOW_EXTRA)

    def __init__(self, cfg):
        """
        :arg cfg: dictionary of bucket settings. The minimum required settings
            must be defined in a voluptuous schema.
            `name` and `type` settings are mandatory. The `type` value for
            each :class:`loudml.bucket.Bucket` is defined in `setup.py` file
            `loudml.buckets` list.

            Voluptuous is used to validate input settings and `cfg` will be
            passed to the validate() function. You can override the `SCHEMA`
            class variable to define new bucket specific settings.
        """
        self._cfg = self.validate(cfg)
        self._pending = []
        self._last_commit = datetime.datetime.now()

    @property
    def timestamp_field(self):
        """
        Return the field name used for date histogram aggregations
        """
        return self.cfg.get('timestamp_field') or 'timestamp'

    @classmethod
    def validate(cls, cfg):
        """Validate configuration against the schema"""
        return schemas.validate(cls.SCHEMA, cfg)

    @property
    def cfg(self):
        """
        Return bucket configuration
        """
        return self._cfg

    @property
    def name(self):
        return self._cfg.get('name')

    @property
    def max_series_per_request(self):
        return self._cfg['max_series_per_request']

    def init(self, *args, **kwargs):
        """
        Perform actions to create a bucket if required. This method
        is optional. Derived classes can omit this function.

        :arg args: any additional arguments will be passed on to the
            derived class instances.

        :arg kwargs: any additional arguments will be passed on to the
            derived class instances.
        """
        pass

    def drop(self):
        """
        Delete all data points contained in the bucket.
        """
        pass

    def nb_pending(self):
        return len(self._pending)

    def clear_pending(self):
        del self._pending[:]

    def commit(self):
        """
        Send data
        """
        if self.nb_pending() > 0:
            self.send_bulk(self._pending)
            self.clear_pending()
        self._last_commit = datetime.datetime.now()

    def must_commit(self):
        """
        Tell if pending data must be sent to the bucket
        """
        nb_pending = self.nb_pending()

        if nb_pending == 0:
            return False
        if nb_pending >= 1000:
            return True
        if (datetime.datetime.now() - self._last_commit).seconds >= 1:
            return True
        return False

    def enqueue(self, req):
        """
        Enqueue query to bulk buffer
        """
        self._pending.append(req)

        if self.must_commit():
            self.commit()

    @abstractmethod
    def send_bulk(self, requests):
        """
        Send write requests to the TSDB.

        :arg requests: a list of requests objects used to
            insert data points in the TSDB.
        """

    @abstractmethod
    def get_times_data(
        self,
        bucket_interval,
        features,
        from_date=None,
        to_date=None,
    ):
        """
        Get data points from the TSDB in the time range [from_date, to_date[

        Note that the time range excludes the final data point.

        This function must return a list of tuples. One tuple is returned
        for each data point in the time range, and contains:
        A time offset, a numpy float array, and a date-time string
        in UTC format:
            (integer, numpy.array, str)

        The numpy.array dimension must be equal to `features` length
        since there is one aggregated float value for each feature.
        All missing values must be filled with `np.nan`, and the helper
        function `np.full` in NumPy is useful to ensure a correct init:
            X = np.full(nb_features, np.nan, dtype=float)

        :arg bucket_interval: the bucket interval in seconds
            to be used for data histogram aggregations.

        :arg features: a list of :class:`loudml.model.Feature`
            features that defines the metrics to aggregate
            in the response

        :arg from_date: minimum time range for TSDB date histogram
            aggregation.

        :arg to_date: maximum time range for TSDB date histogram
            aggregation.
        """

    @abstractmethod
    def insert_data(self, data):
        """
        Insert entry into the index
        """

    @abstractmethod
    def insert_times_data(
        self,
        ts,
        data,
        tags=None,
        *args,
        **kwargs
    ):
        """
        Append data points to the TSDB write requests list,
        with optional tags. This function must call `self.enqueue()`
        with the right arguments.
        Note that the actual write query is performed by `send_bulk`
        function.

        :arg ts: timestamp of the new data point.

        :arg data: a dictionary of str -> float values
            for the new data point

        :arg tags: an optional dictionary of str -> str
            to tag the new data point.

        :arg args: ignored.

        :arg kwargs: any additional arguments received by the Loud ML
            server in POST /buckets/<bucket_name>/_write queries.
        """

    def save_timeseries_prediction(self, prediction, tags=None):
        """
        Save time-series prediction to the bucket
        """
        for bucket in prediction.format_buckets():
            data = bucket['predicted']
            data.update({
                '@{}'.format(key): val
                for key, val in bucket['observed'].items()
            })
            bucket_tags = tags or {}
            stats = bucket.get('stats', None)
            if stats is not None:
                data['score'] = float(stats.get('score'))
                bucket_tags['is_anomaly'] = stats.get('anomaly', False)

            self.insert_times_data(
                ts=bucket['timestamp'],
                tags=bucket_tags,
                data=data,
            )
        self.commit()

    def insert_annotation(
        self,
        dt,
        desc,
        _type,
        _id,
        measurement='annotations',
        tags=None,
    ):
        """
        Insert annotation and return data points saved to the TSDB
        """
        return None

    def update_annotation(
        self,
        dt,
        points,
    ):
        """
        Update annotation in the TSDB
        """
        return None

    def get_top_abnormal_keys(
        self,
        model,
        from_date,
        to_date,
        size=10,
    ):
        raise NotImplementedError()

    def list_anomalies(
        self,
        from_date,
        to_date,
        tags=None,
    ):
        return []


def load_bucket(settings):
    """
    Load bucket
    """
    src_type = settings['type']

    bucket_cls = misc.load_entry_point('loudml.buckets', src_type)
    if bucket_cls is None:
        raise errors.UnsupportedBucket(src_type)
    return bucket_cls(settings)

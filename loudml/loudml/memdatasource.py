import bisect
import logging
import numpy as np

from . import (
    errors,
)
from .misc import (
    ts_to_str,
)
from .datasource import DataSource

class OrderedEntry:
    """
    Ordered index entry
    """

    def __init__(self, value, data=None):
        self.value = value
        self.data = data

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __eq__(self, other):
        return self.value == other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __str__(self):
        return str(self.value)


class Bucket:
    """
    Data bucket
    """

    def __init__(self, key, data=None):
        self.key = key
        self.data = [] if data is None else data

    def format_key(self):
        """
        Return human readable key"
        """
        return str(self.key)


class TimeBucket(Bucket):
    """
    Time-series bucket
    """
    def format_key(self):
        return '%s (%s)' % (self.key, ts_to_str(self.key))


class MemDataSource(DataSource):
    """
    In-memory data source
    """

    def __init__(self, name='mem'):
        super().__init__({
            'name': name,
            'type': 'mem',
        })
        self.data = []

    def insert_data(self, data):
        """
        Insert entry
        """
        self.data.append(data)

    def insert_times_data(self, data):
        """
        Insert time-indexed entry
        """
        bisect.insort(self.data, OrderedEntry(data['timestamp'], data))

    def commit(self):
        pass

    @staticmethod
    def _compute_bucket_avg(bucket, field, default=None):
        """
        Compute metric average
        """

        nb = len(bucket.data)
        if nb:
            values = [entry.data[field] for entry in bucket.data if field in entry.data]
            avg = sum(values) / nb
        else:
            avg = default

        return avg

    @staticmethod
    def _compute_bucket_count(bucket, field):
        """
        Compute metric count
        """
        return sum(field in entry.data for entry in bucket.data)

    def get_times_buckets(
        self,
        from_date=None,
        to_date=None,
        bucket_interval=3600.0,
    ):
        """
        Get buckets of time-series between `from_date` and `to_date`
        """

        lo = bisect.bisect_left(self.data, OrderedEntry(from_date)) if from_date else 0
        bucket_start = from_date

        i = lo
        while bucket_start < to_date:
            bucket_end = bucket_start + bucket_interval
            bucket = TimeBucket(bucket_start)

            for entry in self.data[lo:]:
                if entry.value >= bucket_end:
                    break
                bucket.data.append(entry)
                i += 1
            lo = i

            bucket_start = bucket_end
            yield bucket

    @classmethod
    def _compute_agg_val(cls, bucket, feature):
        """
        Compute aggregation value
        """
        metric = feature.metric
        field = feature.field

        if metric == 'avg':
            agg_val = cls._compute_bucket_avg(bucket, field)
        elif metric == 'count':
            agg_val = cls._compute_bucket_count(bucket, field)
        else:
            logging.error("unknown metric: %s", metric)
            raise errors.UnsupportedMetric(metric)

        if agg_val is None:
            if feature.default is np.nan:
                logging.info(
                    "missing data: field '%s', metric '%s', bucket '%s'",
                    field, metric, bucket.format_key(),
                )
            agg_val = feature.default

        return agg_val

    def get_quadrant_data(
        self,
        model,
        from_date=None,
        to_date=None,
        key=None,
    ):
        raise NotImplemented()

    def get_times_data(
        self,
        model,
        from_date=None,
        to_date=None,
    ):
        features = model.features

        buckets = self.get_times_buckets(
            from_date,
            to_date,
            model.bucket_interval,
        )

        t0 = None

        for bucket in buckets:
            X = np.zeros(model.nb_features, dtype=float)
            timestamp = bucket.key
            timeval = ts_to_str(timestamp)

            for i, feature in enumerate(features):
                X[i] = self._compute_agg_val(bucket, feature)

            if t0 is None:
                t0 = timestamp

            yield (timestamp - t0), X, timeval

    def get_times_start(sel):
        """Get timestamp of first entry"""

        if not self.data:
            raise errors.NoData()
        return self.data[0].value

    def get_times_end(self):
        """Get timestamp of last entry"""

        if not self.data:
            raise errors.NoData()
        return self.data[-1].value

    def save_timeseries_prediction(self, prediction, model):
        raise NotImplemented()

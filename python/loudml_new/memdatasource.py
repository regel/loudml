import bisect
import logging
import numpy as np

from . import (
    errors,
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

    def __init__(self):
        self.data = {}

    def _ensure_index_exists(self, index):
        if index not in self.data:
            self.data[index] = []

    def insert_data(self, index, data):
        """
        Insert entry into the index
        """
        self._ensure_index_exists(index)
        self.data[index].append(data)

    def insert_times_data(self, index, data):
        """
        Insert time-indexed entry
        """
        self._ensure_index_exists(index)
        bisect.insort(self.data[index], OrderedEntry(data['timestamp'], data))

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

    def get_times_buckets(
        self,
        index,
        from_date=None,
        to_date=None,
        bucket_interval=3600.0,
    ):
        """
        Get buckets of time-series between `from_date` and `to_date`
        """

        data = self.data[index]
        lo = bisect.bisect_left(data, OrderedEntry(from_date)) if from_date else 0
        bucket_start = from_date

        i = lo
        while bucket_start < to_date:
            bucket_end = bucket_start + bucket_interval
            bucket = TimeBucket(bucket_start)

            for entry in data[lo:]:
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
        metric = feature['metric']
        field = feature['field']
        nan_is_zero = feature.get('nan_is_zero', False)

        if metric == 'avg':
            agg_val = cls._compute_bucket_avg(bucket, field)
        else:
            logging.error("unknown metric: %s", metric)
            raise errors.UnsupportedMetric()

        if agg_val is None:
            logging.info(
                "missing data: field '%s', metric '%s', bucket '%s'",
                field, metric, bucket.format_key(),
            )
            agg_val = 0 if nan_is_zero else np.nan

        return agg_val

    def get_times_data(
        self,
        model,
        from_date=None,
        to_date=None,
    ):
        features = model.features
        nb_features = len(features)

        buckets = self.get_times_buckets(
            model.index,
            from_date,
            to_date,
            model.bucket_interval,
        )

        t0 = None

        for bucket in buckets:
            X = np.zeros(nb_features, dtype=float)
            timestamp = bucket.key
            timeval = ts_to_str(timestamp)

            i = 0
            for feature in features:
                X[i] = self._compute_agg_val(bucket, feature)
                i += 1

            if t0 is None:
                t0 = timestamp

            yield (timestamp - t0), X, timeval

    def get_times_start(self, index):
        """Get timestamp of first entry"""

        data = self.data.get(index)
        if not data:
            raise errors.NoData()
        return data[0].value

    def get_times_end(self, index):
        """Get timestamp of last entry"""

        data = self.data.get(index)
        if not data:
            raise errors.NoData()
        return data[-1].value

import bisect
import logging
import numpy as np
import gzip
import csv

from . import (
    errors,
)
from .misc import (
    ts_to_str,
    make_ts,
)
from loudml.bucket import Bucket


def make_float(s):
    try:
        val = float(s)
    except ValueError:
        val = s
    return val


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


class DataBucket:
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


class TimeBucket(DataBucket):
    """
    Time-series bucket
    """

    def format_key(self):
        return '%s (%s)' % (self.key, ts_to_str(self.key))


class MemBucket(Bucket):
    """
    In-memory data source
    """

    def __init__(self, name='mem'):
        super().__init__({
            'name': name,
            'type': 'mem',
        })
        self.data = []

    def process_csv_stream(self, fp, timestamp_field, **kwargs):
        reader = csv.DictReader(fp, **kwargs)
        for row in reader:
            timestamp = row[timestamp_field]
            data = {key: make_float(val) for key, val in row.items()}
            data['timestamp'] = make_ts(timestamp)
            self.insert_times_data(data)

    def process_csv(self, path, encoding, timestamp_field, **kwargs):
        logging.info("processing CSV file: %s", path)
        with open(path, 'r', encoding=encoding) as fp:
            self.process_csv_stream(fp, timestamp_field, **kwargs)

    def process_gzip(self, path, encoding, timestamp_field, **kwargs):
        logging.info("processing compressed file: %s", path)
        with gzip.open(path, 'rt', encoding=encoding) as fp:
            self.process_csv_stream(fp, timestamp_field, **kwargs)

    def load_csv(self, path, encoding, timestamp_field, **kwargs):
        if path.endswith('.csv'):
            return self.process_csv(path, encoding, timestamp_field, **kwargs)
        elif path.endswith('.csv.gz'):
            return self.process_gzip(path, encoding, timestamp_field, **kwargs)

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

    def send_bulk(self, requests):
        raise NotImplementedError()

    @staticmethod
    def _compute_bucket_min(bucket, field):
        """
        Compute metric minimum
        """

        nb = len(bucket.data)
        if nb:
            values = [entry.data[field]
                      for entry in bucket.data if field in entry.data]
            _min = min(values)
        else:
            _min = None

        return _min

    @staticmethod
    def _compute_bucket_max(bucket, field):
        """
        Compute metric maximum
        """

        nb = len(bucket.data)
        if nb:
            values = [entry.data[field]
                      for entry in bucket.data if field in entry.data]
            _max = max(values)
        else:
            _max = None

        return _max

    @staticmethod
    def _compute_bucket_avg(bucket, field):
        """
        Compute metric average
        """

        nb = len(bucket.data)
        if nb:
            values = [entry.data[field]
                      for entry in bucket.data if field in entry.data]
            avg = sum(values) / nb
        else:
            avg = None

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
        from_ts = make_ts(from_date)
        to_ts = make_ts(to_date)

        lo = bisect.bisect_left(
            self.data, OrderedEntry(from_ts)) if from_date else 0
        bucket_start = from_ts

        i = lo
        while bucket_start < to_ts:
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
        elif metric == 'min':
            agg_val = cls._compute_bucket_min(bucket, field)
        elif metric == 'max':
            agg_val = cls._compute_bucket_max(bucket, field)
        else:
            logging.error("unknown metric: %s", metric)
            raise errors.UnsupportedMetric(metric)

        return agg_val

    def _get_times_data(
        self,
        features,
        bucket_interval,
        from_date=None,
        to_date=None,
    ):
        buckets = self.get_times_buckets(
            from_date,
            to_date,
            bucket_interval,
        )

        t0 = None

        for bucket in buckets:
            X = np.full(len(features), np.nan, dtype=float)
            timestamp = bucket.key
            timeval = ts_to_str(timestamp)

            for i, feature in enumerate(features):
                agg_val = self._compute_agg_val(bucket, feature)
                if agg_val is None:
                    logging.info(
                        "missing data: field '%s', metric '%s', bucket: %s",
                        feature.field, feature.metric, timeval,
                    )
                else:
                    X[i] = agg_val

            if t0 is None:
                t0 = timestamp

            yield (timestamp - t0), X, timeval

    def get_times_data(
        self,
        bucket_interval,
        features,
        from_date=None,
        to_date=None,
    ):
        return self._get_times_data(
            features,
            bucket_interval,
            from_date,
            to_date,
        )

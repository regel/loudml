"""
InfluxDB module for LoudML
"""

import logging

import influxdb.exceptions
import numpy as np

from influxdb import (
    InfluxDBClient,
)

from . import errors
from loudml.misc import (
    make_ts,
    parse_addr,
    str_to_ts,
)
from loudml.datasource import DataSource

def ts_to_ns(ts):
    """
    Convert second timestamp to integer nanosecond timestamp
    """
    # XXX Due to limited mantis in float numbers, do not multiply directly by 1e9
    return int(int(ts * 1e6) * int(1e3))

def make_ts_ns(mixed):
    """
    Build a nanosecond timestamp from a mixed input (second timestamp or string)
    """
    return ts_to_ns(make_ts(mixed))

def _build_agg(feature):
    """
    Build requested aggregation
    """

    metric = feature.metric
    field = feature.field

    if metric == 'avg':
        agg = "MEAN({})".format(field)
    elif metric == 'count':
        agg = "COUNT({})".format(field)
    else:
        raise errors.NotImplemented(
            "unsupported aggregation '{}'".format(metric),
        )

    return "{} as {}".format(agg, feature.name)

def _build_time_predicates(from_date=None, to_date=None):
    """
    Build time range predicates for 'where' clause
    """

    must = []

    if from_date:
        must.append("time >= {}".format(make_ts_ns(from_date)))
    if to_date:
        must.append("time <= {}".format(make_ts_ns(to_date)))

    return must

def _build_queries(model, from_date=None, to_date=None):
    """
    Build queries according to requested features
    """
    # TODO sanitize inputs to avoid injection!

    must = _build_time_predicates(from_date, to_date)
    where = " where {}".format(" and ".join(must)) if len(must) else ""

    for feature in model.features:
        yield "select {} from {}{} group by time({}s);".format(
            _build_agg(feature),
            feature.measurement,
            where,
            model.bucket_interval,
        )

class InfluxDataSource(DataSource):
    """
    Elasticsearch datasource
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._influxdb = None

    @property
    def addr(self):
        return self.cfg['addr']

    @property
    def db(self):
        return self.cfg['database']

    @property
    def influxdb(self):
        if self._influxdb is None:
            addr = parse_addr(self.addr, default_port=8086)
            logging.info('connecting to influxdb on %s:%d',
                         addr['host'], addr['port'])
            self._influxdb = InfluxDBClient(
                host=addr['host'],
                port=addr['port'],
                database=self.db,
            )

        return self._influxdb

    def create_db(self, db=None):
        """
        Create database
        """
        self.influxdb.create_database(db or self.db)

    def delete_db(self, db=None):
        """
        Delete database
        """
        self.influxdb.drop_database(db or self.db)

    def insert_data(self):
        raise errors.NotImplemented("InfluxDB is a pure time-series database")

    def insert_times_data(self, ts, data, measurement, tags=None):
        """
        Insert data
        """

        ts = make_ts(ts)

        entry = {
            'measurement': measurement,
            'time': ts_to_ns(ts),
            'fields': data,
        }
        if tags:
            entry['tags'] = tags
        self.enqueue(entry)

    def send_bulk(self, requests):
        """
        Send data to InfluxDB
        """
        self.influxdb.write_points(requests)

    def get_times_data(
        self,
        model,
        from_date=None,
        to_date=None,
    ):
        features = model.features
        nb_features = len(features)

        queries = _build_queries(model, from_date, to_date)
        queries = ''.join(queries)

        try:
            results = self.influxdb.query(queries)
        except influxdb.exceptions.InfluxDBClientError as exn:
            raise errors.DataSourceError(self.name, str(exn))

        if not isinstance(results, list):
            results = [results]

        buckets = []

        # Merge results
        for i, result in enumerate(results):
            feature = features[i]

            for j, point in enumerate(result.get_points()):
                agg_val = point.get(feature.name)
                timeval = point['time']

                if j < len(buckets):
                    bucket = buckets[j]
                else:
                    bucket = {
                        'time': timeval,
                        'mod': int(str_to_ts(timeval)) % model.bucket_interval,
                        'values': {},
                    }
                    buckets.append(bucket)

                bucket['values'][feature.name] = agg_val

        # XXX Note that the buckets of InfluxDB results are aligned on
        # modulo(bucket_interval)

        # Build final result
        t0 = None
        for bucket in buckets:
            X = np.zeros(nb_features, dtype=float)
            timeval = bucket['time']
            ts = str_to_ts(timeval)

            for i, feature in enumerate(features):
                agg_val = bucket['values'].get(feature.name)

                if agg_val is None:
                    if feature.default is np.nan:
                        logging.info(
                            "missing data: field '%s', metric '%s', bucket: %s",
                            feature.field, feature.metric, timeval,
                        )
                    agg_val = feature.default

                X[i] = agg_val

            if t0 is None:
                t0 = ts

            yield (ts - t0) / 1000, X, timeval

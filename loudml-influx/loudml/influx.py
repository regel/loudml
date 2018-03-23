"""
InfluxDB module for LoudML
"""

import logging

import influxdb.exceptions
import numpy as np
import requests.exceptions

from voluptuous import (
    Required,
    Schema,
)

from influxdb import (
    InfluxDBClient,
)

from . import (
    errors,
    schemas,
)
from loudml.misc import (
    make_ts,
    parse_addr,
    str_to_ts,
)
from loudml.datasource import DataSource

g_aggregators = {}

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

def aggregator(*aliases):
    """
    Decorator to register aggregators and indexing them by their aliases
    """
    global g_aggregator

    def decorated(func):
        for alias in aliases:
            g_aggregators[alias] = func

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorated

@aggregator('avg', 'mean', 'average')
def _build_avg_agg(feature):
    return "MEAN(\"{}\")".format(feature.field)

@aggregator('count')
def _build_count_agg(feature):
    return "COUNT(\"{}\")".format(feature.field)

@aggregator('deriv', 'derivative')
def _build_derivative_agg(feature):
    return "DERIVATIVE(\"{}\")".format(feature.field)

@aggregator('integral')
def _build_integral_agg(feature):
    return "INTEGRAL(\"{}\")".format(feature.field)

@aggregator('max')
def _build_mode_agg(feature):
    return "MAX(\"{}\")".format(feature.field)

@aggregator('med', 'median')
def _build_median_agg(feature):
    return "MEDIAN(\"{}\")".format(feature.field)

@aggregator('min')
def _build_min_agg(feature):
    return "MIN(\"{}\")".format(feature.field)

@aggregator('mode')
def _build_mode_agg(feature):
    return "MODE(\"{}\")".format(feature.field)

@aggregator('5percentile')
def _build_5percentile_agg(feature):
    return "PERCENTILE(\"{}\", 5)".format(feature.field)

@aggregator('10percentile')
def _build_10percentile_agg(feature):
    return "PERCENTILE(\"{}\", 10)".format(feature.field)

@aggregator('90percentile')
def _build_90percentile_agg(feature):
    return "PERCENTILE(\"{}\", 90)".format(feature.field)

@aggregator('95percentile')
def _build_95percentile_agg(feature):
    return "PERCENTILE(\"{}\", 95)".format(feature.field)

@aggregator('spread')
def _build_spread_agg(feature):
    return "SPREAD(\"{}\")".format(feature.field)

@aggregator('stddev', 'std_dev')
def _build_stddev_agg(feature):
    return "STDDEV(\"{}\")".format(feature.field)

@aggregator('sum')
def _build_sum_agg(feature):
    return "SUM(\"{}\")".format(feature.field)

def _build_agg(feature):
    """
    Build requested aggregation
    """

    global g_aggregators

    aggregator = g_aggregators.get(feature.metric.lower())
    if aggregator is None:
        raise errors.UnsupportedMetric(
            "unsupported aggregation '{}' in feature '{}'".format(
                feature.metric, feature.name,
            ),
        )

    agg = aggregator(feature)
    return "{} as {}".format(agg, feature.name)

def _build_time_predicates(from_date=None, to_date=None):
    """
    Build time range predicates for 'where' clause
    """

    must = []

    if from_date:
        must.append("time >= {}".format(make_ts_ns(from_date)))
    if to_date:
        must.append("time < {}".format(make_ts_ns(to_date)))

    return must

def _build_queries(model, from_date=None, to_date=None):
    """
    Build queries according to requested features
    """
    # TODO sanitize inputs to avoid injection!

    must = _build_time_predicates(from_date, to_date)
    where = " where {}".format(" and ".join(must)) if len(must) else ""

    for feature in model.features:
        yield "select {} from \"{}\"{} group by time({}ms);".format(
            _build_agg(feature),
            feature.measurement,
            where,
            int(model.bucket_interval * 1000),
        )

def catch_query_error(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except (
            influxdb.exceptions.InfluxDBClientError,
            requests.exceptions.RequestException,
        ) as exn:
            raise errors.DataSourceError(self.name, str(exn))
    return wrapper

class InfluxDataSource(DataSource):
    """
    Elasticsearch datasource
    """

    SCHEMA = DataSource.SCHEMA.extend({
        Required('addr'): str,
        Required('database'): schemas.key,
    })

    def __init__(self, cfg):
        cfg['type'] = 'influxdb'
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
            logging.info(
                "connecting to influxdb on %s:%d, using database '%s'",
                addr['host'],
                addr['port'],
                self.db,
            )
            self._influxdb = InfluxDBClient(
                host=addr['host'],
                port=addr['port'],
                database=self.db,
            )

        return self._influxdb

    @catch_query_error
    def create_db(self, db=None):
        """
        Create database
        """
        self.influxdb.create_database(db or self.db)

    @catch_query_error
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

    @catch_query_error
    def send_bulk(self, requests):
        """
        Send data to InfluxDB
        """
        self.influxdb.write_points(requests)

    def get_quadrant_data(
        self,
        model,
        from_date=None,
        to_date=None,
    ):
        raise NotImplemented()

    @catch_query_error
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

        results = self.influxdb.query(queries)

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
        result = []

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

            result.append(((ts - t0) / 1000, X, timeval))

        return result

    def save_timeseries_prediction(self, prediction, model):
        logging.info("saving '%s' prediction to '%s'", model.name, self.name)

        for bucket in prediction.format_buckets():
            self.insert_times_data(
                measurement='prediction_{}'.format(model.name), # Add id? timestamp?
                ts=bucket['timestamp'],
                data=bucket['predicted'],
            )
        self.commit()

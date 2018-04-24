"""
InfluxDB module for LoudML
"""

import logging
import itertools

import influxdb.exceptions
import numpy as np
import requests.exceptions

from voluptuous import (
    Required,
    Optional,
    All,
    Length,
    Boolean,
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
    escape_quotes,
    escape_doublequotes,
    make_ts,
    parse_addr,
    str_to_ts,
    build_agg_name,
)
from loudml.datasource import DataSource


g_max_series_in_partition = 2000
g_aggregators = {}

# Fingerprints code assumes that we return something equivalent to Elasticsearch
def get_metric(name):
    if name.lower() == 'avg':
        return 'avg'
    elif name.lower() == 'mean':
        return 'avg'
    elif name.lower() == 'average':
        return 'avg'
    elif name.lower() == 'stddev':
        return 'std_deviation'
    elif name.lower() == 'std_dev':
        return 'std_deviation'
    elif name.lower() == 'count':
        return 'count'
    elif name.lower() == 'min':
        return 'min'
    elif name.lower() == 'max':
        return 'max'
    elif name.lower() == 'sum':
        return 'sum'
    else:
        return name

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

def format_bool(string):
    if string.lower() == 'true':
        return 'True'
    elif string.lower() == 'false':
        return 'False'
    else:
        return string

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
    return "{} as \"{}\"".format(agg, escape_doublequotes(feature.name))

def _build_count_agg2(feature):
    """
    Build requested aggregation
    """
    agg = _build_count_agg(feature)
    return "{} as \"count_{}\"".format(agg, feature.field)

def _build_sum_agg2(feature):
    """
    Build requested aggregation
    """
    agg = _build_sum_agg(feature)
    return "{} as \"sum_{}\"".format(agg, feature.field)


def _sum_of_squares(feature):
    """
    Build requested aggregation
    """

    return "SUM(\"squares_{}\") as \"sum_squares_{}\"".format(
               feature.field,
               feature.field,
           )

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

def _build_tags_predicates(match_all=None):
    """
    Build tags predicates for 'where' clause
    """
    must = []

    if match_all:
        for item in match_all:
            must.append("\"{}\"='{}'".format(
              escape_doublequotes(item['tag']),
              escape_quotes(format_bool(item['value'])),
            ))

    return must

def _build_key_predicate(tag, val=None):
    """
    Build key predicate for 'where' clause
    """
    must = []

    if val:
        must.append("\"{}\"='{}'".format(
          escape_doublequotes(tag),
          escape_quotes(format_bool(val)),
        ))

    return must

def _build_queries(model, from_date=None, to_date=None):
    """
    Build queries according to requested features
    """
    # TODO sanitize inputs to avoid injection!

    time_pred = _build_time_predicates(from_date, to_date)

    for feature in model.features:
        must = time_pred + _build_tags_predicates(feature.match_all)

        where = " where {}".format(" and ".join(must)) if len(must) else ""

        yield "select {} from \"{}\"{} group by time({}ms);".format(
            _build_agg(feature),
            escape_doublequotes(feature.measurement),
            where,
            int(model.bucket_interval * 1000),
        )

def _build_quad(model, agg, from_date=None, to_date=None, key_val=None, limit=0, offset=0):
    """
    Build aggregation query according to requested features
    """
    # TODO sanitize inputs to avoid injection!

    time_pred = _build_time_predicates(from_date, to_date)

    must = time_pred + _build_tags_predicates(agg.match_all) \
           + _build_key_predicate(model.key, key_val)

    where = " where {}".format(" and ".join(must)) if len(must) else ""

    yield "select {} from \"{}\"{} group by {},time({}ms) fill(0) slimit {} soffset {};".format(
        ','.join(list(set([_build_agg(feature) for feature in agg.features] + \
                          [_build_count_agg2(feature) for feature in agg.features] + \
                          [_build_sum_agg2(feature) for feature in agg.features]))),
        escape_doublequotes(agg.measurement),
        where,
        model.key,
        int(model.daytime_interval * 1000),
        limit,
        offset,
    )
    sum_of_squares = []
    for feature in agg.features:
        if feature.metric == 'stddev':
            sum_of_squares.append(feature)

    if len(sum_of_squares) > 0:
        yield "select {} from ( select \"{}\"*\"{}\" as \"squares_{}\" from \"{}\"{} ) where {} group by {},time({}ms) fill(0) slimit {} soffset {};".format(
            ','.join(list(set([_sum_of_squares(feature) for feature in sum_of_squares]))),
            escape_doublequotes(feature.field),
            escape_doublequotes(feature.field),
            escape_doublequotes(feature.field),
            escape_doublequotes(agg.measurement),
            where,
            " and ".join(time_pred),
            model.key,
            int(model.daytime_interval * 1000),
            limit,
            offset,
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
        Optional('dbuser'): All(schemas.key, Length(max=256)),
        Optional('dbuser_password'): str,
        Optional('use_ssl', default=False): Boolean(),
        Optional('verify_ssl', default=False): Boolean(),
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
    def dbuser(self):
        return self.cfg.get('dbuser')

    @property
    def dbuser_password(self):
        return self.cfg.get('dbuser_password')

    @property
    def use_ssl(self):
        return self.cfg.get('use_ssl') or False

    @property
    def verify_ssl(self):
        return self.cfg.get('verify_ssl') or False

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
                username=self.dbuser,
                password=self.dbuser_password,
                ssl=self.use_ssl,
                verify_ssl=self.verify_ssl,
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

    def insert_times_data(self, ts, data, measurement='generic', tags=None, timestamp_field=None):
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

    @catch_query_error
    def _get_quadrant_data(
        self,
        model,
        agg,
        from_date=None,
        to_date=None,
        key=None,
        limit=0,
        offset=0,
    ):
        queries = _build_quad(model, agg, from_date, to_date, key, limit, offset)
        queries = ''.join(queries)
        results = self.influxdb.query(queries)

        if not isinstance(results, list):
            results = [results]

        buckets_dict = dict()
        for i, result in enumerate(results):
            for (measurement, tags), points in result.items():
                key = tags[model.key]
                if key in buckets_dict:
                    buckets = buckets_dict[key]
                else:
                    buckets = []

                for j, point in enumerate(points):
                    timeval = point['time']
                    if j < len(buckets):
                        bucket = buckets[j]
                    else:
                        bucket = {
                            'key_as_string': timeval,
                        }
                        for feature in agg.features:
                            agg_name = build_agg_name(agg.measurement, feature.field)
                            bucket[agg_name] = {
                                'count': 0.0,
                                'min': 0.0,
                                'max': 0.0,
                                'avg': 0.0,
                                'sum': 0.0,
                                'sum_of_squares': 0.0,
                                'variance': 0.0,
                                'std_deviation': 0.0,
                            }
                        buckets.append(bucket)

                    for feature in agg.features:
                        agg_name = build_agg_name(agg.measurement, feature.field)
                        agg_val = point.get(feature.name)
                        if agg_val is not None:
                            bucket[agg_name][get_metric(feature.metric)] = float(agg_val)

                        agg_val = point.get("count_{}".format(feature.field))
                        if agg_val is not None:
                            bucket[agg_name]['count'] = float(agg_val)

                        agg_val = point.get("sum_{}".format(feature.field))
                        if agg_val is not None:
                            bucket[agg_name]['sum'] = float(agg_val)

                        agg_val = point.get("sum_squares_{}".format(feature.field))
                        if agg_val is not None:
                            bucket[agg_name]['sum_of_squares'] = float(agg_val)
    
                if len(results) == (i+1):
                    yield(key, buckets)
                else:
                    buckets_dict[key] = buckets


    def get_quadrant_data(
        self,
        model,
        agg,
        from_date=None,
        to_date=None,
        key=None,
    ):
        global g_max_series_in_partition
#        result = self.influxdb.query("SHOW SERIES CARDINALITY")
        result = self.influxdb.query("SHOW TAG VALUES CARDINALITY WITH KEY = \"{}\"".format(model.key))
        for (_, tags), points in result.items():
            point = next(points)
            total_series = int(point['count'])

        output = itertools.chain()
        gens = []
        for offset in range(int(total_series / g_max_series_in_partition) + 1):
            gens.append(self._get_quadrant_data(model=model,
                                    agg=agg,
                                    from_date=from_date,
                                    to_date=to_date,
                                    key=key,
                                    limit=g_max_series_in_partition,
                                    offset=offset*g_max_series_in_partition))
        for gen in gens:
            output = itertools.chain(output, gen)
        return output


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

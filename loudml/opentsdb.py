"""
OpenTSDB module for Loud ML.

 * OpenTSDBClient class handles connection to OpenTSDB database
 * OpenTSDBBucket class is capable to read and write timed data
"""

import logging
import requests
import json
import gzip

import numpy as np

from voluptuous import (
    Required,
    Optional,
    Boolean,
)

from . import (
    errors,
)
from loudml.misc import (
    make_ts,
    parse_addr,
)
from loudml.bucket import Bucket


def floor(ts, interval):
    return int(ts / interval) * interval


def format_bool(o):
    if str(o).lower() == 'true':
        return 'true'
    elif str(o).lower() == 'false':
        return 'false'
    else:
        return o


# Available aggregators on OpenTSDB: http://localhost:4242/api/aggregators
AGGREGATORS = {
    'avg': 'avg',
    'mean': 'avg',
    'average': 'avg',
    'count': 'sum',  # sum of downsampled bucket counters
    'min': 'min',
    'max': 'max',
    'sum': 'sum',
}
DOWNSAMPLE = {
    'avg': 'avg',
    'mean': 'avg',
    'average': 'avg',
    'stddev': 'dev',
    'std_dev': 'dev',
    'count': 'count',
    'min': 'min',
    'max': 'max',
    'sum': 'sum',
    '90percentile': 'p90',
    '95percentile': 'p95'
}


def _build_time_predicates(
    from_date=None,
    to_date=None,
):
    """
    Build time range predicates
    """
    must = []

    if from_date:
        must.append("start={}ms".format(
            int(make_ts(from_date)*1e3)))
    if to_date:
        must.append("end={}ms".format(
            int(make_ts(to_date)*1e3)))

    return "&".join(must)


def _build_tags_predicates(match_all=None):
    """
    Build tags predicates
    """
    must = {}

    if match_all:
        for condition in match_all:
            must[condition['tag']] = condition['value']

    return must


def catch_query_error(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except (
            requests.exceptions.RequestException,
            requests.exceptions.ConnectionError
        ) as exn:
            raise errors.BucketError(self.name, str(exn))
    return wrapper


class OpenTSDBResult(object):
    """
    Helper class to parse query result
    """

    def __init__(self, response):
        self._response = response

    def __repr__(self):
        return "OpenTSDB results: {}...".format(str(self._response)[:200])

    def get_points(self):
        if not self._response:
            return []

        return self._response[0]['dps'].items()


class OpenTSDBClient(object):
    """
    Client for OpenTSDB database
    """

    def __init__(
        self,
        host="localhost",
        port=4242,
        ssl=False,
        verify_ssl=False,
        ssl_cert_path='',
        user='',
        password=''
    ):
        """
        Set proper schema based on SSL param, open session
        and supply basic authentication creds if given
        https://requests.readthedocs.io/en/latest/user/advanced/#ssl-cert-verification
        """
        schema = "http"
        if ssl:
            schema = "https"

        self.url = "%s://%s:%d" % (schema, host, port)
        self.session = requests.Session()
        if user and password:
            self.session.auth = (user, password)
        if ssl_cert_path:
            self.session.verify = ssl_cert_path

    def drop(self, tag_to_drop):
        """
        Set tsd.http.query.allow_delete = true in opentsdb.conf
        http://opentsdb.net/docs/build/html/user_guide/configuration.html
        """
        query_url = "%s/api/suggest?type=metrics" % self.url
        resp = self.session.get(query_url)
        if not resp.ok:
            logging.error(
                'OpenTSDB error',
                resp.status_code, resp.text[:200])

        resp.raise_for_status()
        metrics = resp.json()
        for metric_name in metrics:
            query_url = "{}/api/query?start=0&m=sum:{}{}".format(
                self.url,
                metric_name,
                self._format_tags({'loudml': tag_to_drop})
            )
            try:
                resp = self.session.delete(query_url)
            except requests.exceptions.SSLError as exn:
                logging.error('OpenTSDB SSL error', str(exn))

    def query(self, queries):
        """
        Run a list of queries against OpenTSDB
        """
        if not isinstance(queries, list):
            queries = [queries]

        results = []

        for q in queries:
            time_pred = _build_time_predicates(q["start"], q["end"])

            query_url = "{}/api/query?{}&m={}:{}:{}{}".format(
                self.url,
                time_pred,
                AGGREGATORS.get(q["metric"], 'first'),
                q["down_sampler"],
                q["field"],
                self._format_tags(q["tags"])
            )
            # TODO: OpenTSDB is capable of running multiple subquries in
            # one shot. Refactor it to use one request to server
            try:
                resp = self.session.get(query_url)
            except requests.exceptions.SSLError as exn:
                logging.error('OpenTSDB SSL error', str(exn))

            if resp.ok:
                results.append(OpenTSDBResult(resp.json()))
            elif "No such name for" in resp.text:
                # Known OpenTSDB issue with 400 and trace if metric not found
                # https://github.com/OpenTSDB/opentsdb/issues/792
                results.append(OpenTSDBResult([]))
            else:
                logging.error(
                    'OpenTSDB error',
                    resp.status_code, resp.text[:200])

        return results

    def _format_tags(self, tags):
        """
        Formats tags dict into query like {tag=value,...}
        """
        res = ",".join([
            "{}={}".format(k, format_bool(v))
            for k, v in tags.items()
        ])
        return "{" + res + "}"

    def put(self, points):
        """
        Store points in database and sync
        """
        url = "%s/api/put?sync" % self.url

        try:
            additional_headers = {}
            additional_headers['Content-Encoding'] = 'gzip'
            additional_headers['Content-type'] = 'application/json'
            encoded = json.dumps(points).encode('utf-8')
            request_body = gzip.compress(encoded)
            resp = self.session.post(
                url, data=request_body, headers=additional_headers)
        except requests.exceptions.SSLError as exn:
            logging.error('OpenTSDB SSL error', str(exn))

        if not resp.ok:
            logging.error('OpenTSDB error', resp.status_code, resp.text[:200])

        return resp


class OpenTSDBBucket(Bucket):
    """
    OpenTSDB bucket
    """

    SCHEMA = Bucket.SCHEMA.extend({
        Optional('global_tag', default='loudml'): str,
        Required('addr'): str,
        Optional('user', default=""): str,
        Optional('password', default=""): str,
        Optional('use_ssl', default=False): Boolean(),
        # Useful if the server is using self-signed certificates
        Optional('verify_ssl', default=False): Boolean(),
        Optional('ssl_cert_path', default=""): str,
    })

    def __init__(self, cfg):
        cfg['type'] = 'opentsdb'
        super().__init__(cfg)
        self._opentsdb = None

    @property
    def global_tag(self):
        return self.cfg['global_tag']

    @property
    def addr(self):
        return self.cfg['addr']

    @property
    def user(self):
        return self.cfg['user'] or ''

    @property
    def password(self):
        return self.cfg['password'] or ''

    @property
    def use_ssl(self):
        return self.cfg.get('use_ssl') or False

    @property
    def verify_ssl(self):
        return self.cfg.get('verify_ssl') or False

    @property
    def ssl_cert_path(self):
        return self.cfg.get('ssl_cert_path') or ''

    @property
    def opentsdb(self):
        if self._opentsdb is None:
            addr = parse_addr(self.addr, default_port=4242)
            logging.info(
                "connecting to opentsdb on %s:%d",
                addr['host'],
                addr['port'],
            )
            self._opentsdb = OpenTSDBClient(
                host=addr['host'],
                port=addr['port'],
                ssl=self.use_ssl,
                verify_ssl=self.verify_ssl,
                ssl_cert_path=self.ssl_cert_path,
                user=self.user,
                password=self.password
            )

        return self._opentsdb

    @catch_query_error
    def drop(self, db=None):
        self.opentsdb.drop(self.global_tag)

    def insert_data(self, data):
        raise errors.NotImplemented("OpenTSDB is a pure time-series database")

    def insert_times_data(
        self,
        ts,
        data,
        measurement=None,
        tags={},
        sync=False,
        *args,
        **kwargs
    ):
        """
        Insert data
        """
        filtered = filter(lambda item: item[1] is not None, data.items())
        tags = tags.copy()
        tags['loudml'] = self.global_tag
        millis = int(make_ts(ts)*1e3)
        for k, v in filtered:
            self.enqueue({
                'metric': k,
                'timestamp': millis,
                'value': v,
                'tags': tags,
            })

    # @catch_query_error
    def send_bulk(self, requests):
        """
        Send data to OpenTSDB
        """
        self.opentsdb.put(requests)

    def _build_times_queries(
        self,
        bucket_interval,
        features,
        from_date=None,
        to_date=None,
    ):
        """
        Build queries according to requested features

        Notes:
         * OpenTSDB requires int timestamps
         * tags is required param

        http://opentsdb.net/docs/build/html/api_http/put.html
        """
        start = floor(make_ts(from_date), bucket_interval)
        end = floor(
            make_ts(to_date), bucket_interval) - bucket_interval
        queries = []
        for feature in features:
            queries.append({
                "start": int(start),
                "end": int(end),
                "metric": feature.metric,
                "down_sampler": "{}s-{}-nan".format(
                    int(bucket_interval),
                    DOWNSAMPLE.get(feature.metric, 'avg')),
                "field": feature.field,
                "tags": _build_tags_predicates(feature.match_all)
            })
        return queries

    # @catch_query_error
    def get_times_data(
        self,
        bucket_interval,
        features,
        from_date=None,
        to_date=None,
    ):
        """
        Queries OpenTSDB based on metric and params
        """
        nb_features = len(features)

        queries = self._build_times_queries(
            bucket_interval, features, from_date, to_date)

        results = self.opentsdb.query(queries)
        if not isinstance(results, list):
            results = [results]

        buckets = []
        # Merge results
        for i, result in enumerate(results):
            feature = features[i]

            for j, point in enumerate(result.get_points()):
                agg_val = point[1] if point[1] != 'NaN' else None
                if agg_val is None and feature.metric == 'count':
                    agg_val = 0
                timeval = point[0]

                if j < len(buckets):
                    bucket = buckets[j]
                else:
                    bucket = {
                        'time': int(timeval),
                        'values': {},
                    }
                    buckets.append(bucket)

                bucket['values'][feature.name] = agg_val

        # Build final result
        t0 = None
        result = []

        for bucket in buckets:
            X = np.full(nb_features, np.nan, dtype=float)
            ts = bucket['time']

            for i, feature in enumerate(features):
                agg_val = bucket['values'].get(feature.name)
                if agg_val is None:
                    logging.info(
                        "missing data: field '%s', metric '%s', bucket: %s",
                        feature.field, feature.metric, ts,
                    )
                else:
                    X[i] = agg_val

            if t0 is None:
                t0 = ts

            result.append(((ts - t0), X, ts))

        return result

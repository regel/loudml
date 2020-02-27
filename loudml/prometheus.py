"""
OpenTSDB module for Loud ML.

 * OpenTSDBClient class handles connection to OpenTSDB database
 * OpenTSDBBucket class is capable to read and write timed data
"""

import logging
import requests
import json

import numpy as np

from voluptuous import (
    Required,
    Optional,
    Boolean,
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
    ts_to_str,
)
from loudml.bucket import Bucket


class PrometheusClient(object):
    """
    Client for Prometheus
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

    def query(self, queries):
        """
        Run a list of queries against Prometheus
        """

    def put(self, entry):
        """
        Store a point in database
        """


class PrometheusBucket(Bucket):
    """
    Prometheus bucket
    """

    SCHEMA = Bucket.SCHEMA.extend({
        Required('addr'): str,
        Optional('user', default=""): str,
        Optional('password', default=""): str,
        Optional('use_ssl', default=False): Boolean(),
        # Useful if the server is using self-signed certificates
        Optional('verify_ssl', default=False): Boolean(),
        Optional('ssl_cert_path', default=""): str,
    })

    def __init__(self, cfg):
        cfg['type'] = 'prometheus'
        super().__init__(cfg)
        self._prometheus = None

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
    def prometheus(self):
        if self._prometheus is None:
            addr = parse_addr(self.addr, default_port=4242)
            logging.info(
                "connecting to prometheus on %s:%d",
                addr['host'],
                addr['port'],
            )
            self._prometheus = PrometheusClient(
                host=addr['host'],
                port=addr['port'],
                ssl=self.use_ssl,
                verify_ssl=self.verify_ssl,
                ssl_cert_path=self.ssl_cert_path,
                user=self.user,
                password=self.password
            )

        return self._prometheus

    def insert_data(self, data):
        raise errors.NotImplemented("Prometheus is a pure time-series database")

    def insert_times_data(
        self,
        ts,
        data,
        measurement=None,
        tags={},
        *args,
        **kwargs
    ):
        """
        Insert data
        """
        ts = int(make_ts(ts))
        filtered = filter(lambda item: item[1] is not None, data.items())
        for k, v in filtered:
            entry = {
                'metric': k,
                'timestamp': ts,
                'value': v,
                'tags': tags,
                'sync': sync
            }
            self.enqueue(entry)

    def send_bulk(self, requests):
        """
        Send data to Prometheus
        """
        for entry in requests:
            self.prometheus.put(entry)

    def _build_times_queries(
        self,
        bucket_interval,
        features,
        from_date=None,
        to_date=None,
    ):
        """
        Build queries according to requested features
        """
        queries = []

        # TODO: put black magic here

        return queries

    def get_times_data(
        self,
        bucket_interval,
        features,
        from_date=None,
        to_date=None,
    ):
        """
        Queries Prometheus based on metric and params
        """
        nb_features = len(features)

        queries = self._build_times_queries(
            bucket_interval, features, from_date, to_date)

        results = self.prometheus.query(queries)

        if not isinstance(results, list):
            results = [results]

        buckets = []
        # Merge results
        for i, result in enumerate(results):
            feature = features[i]

            for j, point in enumerate(result.get_points()):
                agg_val = point[1]
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

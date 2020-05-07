"""
Prometheus module for Loud ML.

 * PrometheusClient class handles connection to Prometheus
 * PrometheusBucket class is capable to read and write timed data
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

from loudml import (
    errors,
    schemas,
)
from loudml.requests import (
    perform_request,
    DEFAULT_REQUEST_TIMEOUT
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




# Prometheus aggregation function:
# https://prometheus.io/docs/prometheus/latest/querying/operators/#aggregation-operators
# sum (calculate sum over dimensions)
# min (select minimum over dimensions)
# max (select maximum over dimensions)
# avg (calculate the average over dimensions)
# stddev (calculate population standard deviation over dimensions)
# stdvar (calculate population standard variance over dimensions)
# count (count number of elements in the vector)
# count_values (count number of elements with the same value)
# bottomk (smallest k elements by sample value)
# topk (largest k elements by sample value)
# quantile (calculate φ-quantile (0 ≤ φ ≤ 1) over dimensions)

AGGREGATORS = {
    'avg': 'avg({}{})',
    'mean': 'avg({}{})',
    'average': 'avg({}{})',
    'stddev': 'stddev({}{})',
    'std_dev': 'stddev({}{})',
    'count': 'count({}{})',
    'min': 'min({}{})',
    'max': 'max({}{})',
    'sum': 'sum({}{})',
    'bottomk': 'bottomk(1,{}{})', # TODO: need to handle bottomk(X) param
    'topk': 'topk(1,{}{})', # TODO: need to handle topk(X) param
    'deriv': None,
    'derivative': None,
    'integral': None,
    'med': None,
    'median': None,
    'mode': None,
    '5percentile': 'quantile(0.05,{}{})',
    '10percentile': 'quantile(0.10,{}{})',
    '90percentile': 'quantile(0.90,{}{})',
    '95percentile': 'quantile(0.95,{}{})'
}


def _build_tags_predicates(match_all=None):
    """
    Build tags predicates
    """
    must = []

    if match_all:
        for condition in match_all:
            must.append('{}="{}"'.format(condition['tag'], condition['value']))

    return "{" + ",".join(must) + "}"


class PrometheusResult(object):
    """
    Helper class to parse query result
    """

    def __init__(self, response):
        self._response = response

    def __repr__(self):
        return "Prometheus results: {}...".format(str(self._response)[:200])

    def get_points(self):
        if ((not self._response)
            or (not 'data' in self._response)
            or (not 'result' in self._response['data'])
            or (not len(self._response['data']['result']))
        ):
            return []
        # values are pairs of [timestamp, value]
        return self._response['data']['result'][0]['values']


class PrometheusClient(object):
    """
    Client for Prometheus
    """

    def __init__(
        self,
        host="localhost",
        port=9090,
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


    def build_url_params(self, q):
        """
        Forms a query URL params from bits.
        TODO: add better aggregator functions handling
        """
        params = {
            'start': int(make_ts(q['start'])),
            'end': int(make_ts(q['end'])),
            'step': q["step"],
            'timeout': DEFAULT_REQUEST_TIMEOUT
        }

        aggregator = AGGREGATORS.get(q["aggregator"])
        if aggregator:
            params['query'] = aggregator.format(q["metric_name"], q["tags"])
        else:
            logging.warning('Unsupported aggregation operator.'
                'Please submit a ticket on GitHub :)')
            params['query'] = "{}{}".format(q["metric_name"], q["tags"])

        return params


    def query(self, queries):
        """
        Run a list of queries against Prometheus.
        API request should look like:
        # http://0.0.0.0:9090/api/v1/query_range?query=go_memstats_alloc_bytes{instance=%22localhost:9090%22,%20job=%22prometheus%22}&start=1584783120&end=1584786720&step=15
        # http://0.0.0.0:9090/api/v1/query_range?query=avg(go_memstats_alloc_bytes{instance=%22localhost:9090%22,%20job=%22prometheus%22})&start=1584783120&end=1584786720&step=15
        # quantile:
        http://0.0.0.0:9090/api/v1/query_range?query=quantile(0.05,%20go_memstats_alloc_bytes{instance=%22localhost:9090%22,%20job=%22prometheus%22})&start=1586681165&end=1586681765&step=1
        """
        if not isinstance(queries, list):
            queries = [queries]

        results = []

        for q in queries:
            params = self.build_url_params(q)
            try:
                resp = perform_request(
                    self.url,
                    'GET',
                    '/api/v1/query_range',
                    session=self.session,
                    params=params,
                    body=None,
                    timeout=DEFAULT_REQUEST_TIMEOUT,
                    ignore=(),
                    headers=None,
                )
            except requests.exceptions.SSLError as exn:
                logging.error('Prometheus SSL error', str(exn))

            if resp.ok:
                results.append(PrometheusResult(resp.json()))
            else:
                logging.error('Prometheus error',
                    resp.status_code, resp.text[:200])

        return results


    def put(self, entry):
        """
        Store a point in database
        """
        raise NotImplementedError("Prometheus bucket: not implemented yet")


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
        if not self._prometheus:
            addr = parse_addr(self.addr, default_port=9090)
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
        raise NotImplementedError("Prometheus is a pure time-series database")

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
            self.enqueue({
                'metric': k,
                'timestamp': ts,
                'value': v,
                'tags': tags,
            })

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
        for feature in features:
            queries.append({
                "start": int(make_ts(from_date)),
                "end": int(make_ts(to_date)),
                "aggregator": feature.metric,
                "step": int(bucket_interval),
                "metric_name": feature.field,
                "tags": _build_tags_predicates(feature.match_all)
            })
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

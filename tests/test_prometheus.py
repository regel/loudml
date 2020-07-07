from loudml.model import Model
from loudml.prometheus import (
    _build_tags_predicates,
    PrometheusBucket
)
from loudml.misc import (
    nan_to_none,
    make_ts,
)
from loudml.requests import (
    DEFAULT_REQUEST_TIMEOUT
)

import datetime
import logging
import os
import unittest

logging.getLogger('tensorflow').disabled = True


FEATURES = [
    {
        'name': 'avg_foo',
        'metric': 'avg',
        'measurement': 'measure1',
        'field': 'foo',
        'default': 0,
    }
]

if 'PROMETHEUS_ADDR' in os.environ:
    ADDR = os.environ['PROMETHEUS_ADDR']
else:
    ADDR = 'localhost:9090'


class TestPrometheusQuick(unittest.TestCase):
    def setUp(self):
        bucket_interval = 3

        t0 = int(datetime.datetime.now().timestamp())
        t0 -= t0 % bucket_interval

        self.t0 = t0

        self.source = PrometheusBucket({
            'name': 'test',
            'addr': ADDR,
        })

        self.model = Model(dict(
            name="test-model",
            offset=30,
            span=300,
            bucket_interval=3,
            interval=60,
            features=FEATURES,
        ))

    def tearDown(self):
        self.source.drop()

    def test_build_tags_predicates(self):
        self.assertEqual(
            _build_tags_predicates(), '{}',
        )

        self.assertEqual(
            _build_tags_predicates([
                {'tag': 'foo', 'value': 'bar'},
                {'tag': 'a', 'value': 'b'},
                {'tag': 'int', 'value': 42},
                {'tag': 'bool', 'value': True},
            ]), '{foo="bar",a="b",int="42",bool="True"}'
        )

    def test_build_times_queries(self):
        queries = self.source._build_times_queries(
            bucket_interval=self.model.bucket_interval,
            features=self.model.features,
            from_date=1515404366.1234,
            to_date="2018-01-08T14:59:25.456Z",
        )

        self.assertEqual(len(queries), 1)

        self.assertEqual(
            queries[0],
            {
                'start': 1515404366,
                'end': 1515423565,
                'aggregator': 'avg',
                'step': 3,
                'metric_name': 'foo',
                'tags': '{}'
            }
        )

    def test_build_query_url_params(self):
        query = {
            "start": 42,
            "end": 42,
            "aggregator": "95percentile",
            "step": 15,
            "metric_name": "foo",
            "tags": "{}"
        }
        params = self.source.prometheus.build_url_params(query)
        self.assertEqual(
            params,
            {
                "start": 42,
                "end": 42,
                "step": 15,
                "query": "quantile(0.95,foo{})",
                "timeout": DEFAULT_REQUEST_TIMEOUT
            }
        )

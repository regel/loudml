import loudml.vendor  # noqa
from loudml.model import Model
from loudml.opentsdb import (
    _build_time_predicates,
    _build_tags_predicates,
    OpenTSDBBucket,
)
from loudml.misc import (
    nan_to_none,
    make_ts,
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

if 'OPENTSDB_ADDR' in os.environ:
    ADDR = os.environ['OPENTSDB_ADDR']
else:
    ADDR = 'localhost:4242'


class TestOpenTSDBQuick(unittest.TestCase):
    def setUp(self):
        bucket_interval = 3

        t0 = int(datetime.datetime.now().timestamp())
        t0 -= t0 % bucket_interval

        self.t0 = t0

        self.source = OpenTSDBBucket({
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

        self.source.insert_times_data(
            measurement='measure1',
            ts=t0+3,
            data={
                'foo': 42,
            },
            tags={'host': 'localhost'},
            sync=True
        )

        self.source.commit()

    def tearDown(self):
        # TODO: clear test data somehow, drop() does nothing right now
        self.source.drop()

    def test_build_time_predicates(self):
        self.assertEqual(
            _build_time_predicates(), "",
        )
        self.assertEqual(
            _build_time_predicates(
                from_date=1515404366, to_date="2018-01-08T14:59:25.456Z",
            ),
            "start=1515404366&end=2018-01-08T14:59:25.456Z",
        )

    def test_build_tags_predicates(self):
        self.assertEqual(
            _build_tags_predicates(), {},
        )
        self.assertEqual(
            _build_tags_predicates([
                {'tag': 'foo', 'value': 'bar'},
                {'tag': 'a "', 'value': 'b \''},
                {'tag': 'int', 'value': 42},
                {'tag': 'bool', 'value': True},
            ]), {
                'foo': 'bar',
                'a "': 'b \'',
                'int': 42,
                'bool': True,
            }
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
                'down_sampler': '3s-avg-nan',
                'metric_name': 'foo',
                'tags': {}
            }
        )

    def test_get_times_data(self):
        logging.info("[%d %d]", self.t0, self.t0)
        res = self.source.get_times_data(
            bucket_interval=self.model.bucket_interval,
            features=self.model.features,
            from_date=self.t0,
            to_date=self.t0+8,
        )

        foo_avg = []

        for line in res:
            foo_avg.append(nan_to_none(line[1][0]))

        self.assertEqual(foo_avg, [42.0, 42.0, None])

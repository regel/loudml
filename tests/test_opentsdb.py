from loudml.model import Model
from loudml.opentsdb import (
    _build_time_predicates,
    _build_tags_predicates,
    OpenTSDBBucket,
)
from loudml.membucket import MemBucket
from loudml.misc import (
    nan_to_none,
    make_ts,
)

import copy
import datetime
import logging
import numpy as np
import os
import unittest

logging.getLogger('tensorflow').disabled = True


FEATURES = [
    {
        'name': 'avg_foo',
        'metric': 'avg',
        'field': 'foo',
        'default': 0,
    },
    {
        'name': 'count_bar',
        'metric': 'count',
        'field': 'bar',
        'default': 0,
    },
    {
        'name': 'avg_baz',
        'metric': 'avg',
        'field': 'baz',
        'match_all': [
            {'tag': 'mytag', 'value': 'myvalue'},
        ],
        'default': 0,
    },
]

FEATURES_MATCH_ALL_TAG1 = [
    {
        'name': 'avg_baz',
        'metric': 'avg',
        'field': 'baz',
        'match_all': [
            {'tag': 'tag_kw', 'value': 'tag1'},
        ],
    },
]
FEATURES_MATCH_ALL_TAG2 = [
    {
        'name': 'avg_baz',
        'metric': 'avg',
        'field': 'baz',
        'match_all': [
            {'tag': 'tag_kw', 'value': 'tag2'},
            {'tag': 'tag_int', 'value': 7},
            {'tag': 'tag_bool', 'value': True},
        ],
    },
]

if 'OPENTSDB_ADDR' in os.environ:
    ADDR = os.environ['OPENTSDB_ADDR']
else:
    ADDR = 'localhost:4242'


class TestOpenTSDBQuick(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bucket_interval = 3

        t0 = int(datetime.datetime.now().timestamp())
        t0 -= t0 % bucket_interval

        cls.t0 = t0

        cls.source = OpenTSDBBucket({
            'name': 'test',
            'addr': ADDR,
        })
        cls.source.drop()
        cls.source.init()

        cls.model = Model(dict(
            name="test-model",
            offset=30,
            span=300,
            bucket_interval=3,
            interval=60,
            features=FEATURES,
        ))

        data = [
            # (foo, bar, timestamp)
            (1, 33, t0 - 1),  # excluded
            (2, 120, t0), (3, 312, t0 + 1),
            # empty
            (4, 18, t0 + 7),
            (5, 78, t0 + 9),  # excluded
        ]
        for foo, bar, ts in data:
            cls.source.insert_times_data(
                ts=ts,
                data={
                    'foo': foo,
                }
            )
            cls.source.insert_times_data(
                ts=ts,
                data={
                    'bar': bar,
                }
            )
            cls.source.insert_times_data(
                ts=ts,
                tags={
                    'tag_kw': 'tag1',
                    'tag_int': 9,
                    'tag_bool': False,
                },
                data={
                    'baz': bar,
                }
            )
            cls.source.insert_times_data(
                ts=ts,
                tags={
                    'tag_kw': 'tag2',
                    'tag_int': 7,
                    'tag_bool': True,
                },
                data={
                    'baz': -bar,
                }
            )

        cls.source.commit()

    @classmethod
    def tearDownClass(cls):
        cls.source.drop()

    def test_build_time_predicates(self):
        self.assertEqual(
            _build_time_predicates(), "",
        )
        self.assertEqual(
            _build_time_predicates(
                from_date=1515404367, to_date="2018-01-08T14:59:27.456Z",
            ),
            'start=1515404367000ms&end=1515423567456ms',
        )

    def test_build_tags_predicates(self):
        self.assertDictEqual(
            _build_tags_predicates(), {},
        )
        self.assertDictEqual(
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
            from_date=1515404367.1234,
            to_date="2018-01-08T14:59:27.456Z",
        )

        self.assertEqual(len(queries), 3)

        self.assertDictEqual(
            queries[0],
            {
                'start': 1515404367,
                'end': 1515423564,
                'metric': 'avg',
                'down_sampler': '3s-avg-nan',
                'field': 'foo',
                'tags': {}
            }
        )

    def test_get_times_data(self):
        res = self.source.get_times_data(
            bucket_interval=self.model.bucket_interval,
            features=self.model.features,
            from_date=self.t0,
            to_date=self.t0 + 9,
        )

        foo_avg = []
        bar_count = []

        for line in res:
            foo_avg.append(nan_to_none(line[1][0]))
            bar_count.append(nan_to_none(line[1][1]))

        self.assertEqual(foo_avg, [2.5, None, 4.0])
        self.assertEqual(bar_count, [2.0, 0, 1.0])

    def test_get_times_data2(self):
        res = self.source.get_times_data(
            bucket_interval=self.model.bucket_interval,
            features=self.model.features,
            from_date=self.t0,
            to_date=self.t0 + 9,
        )

        # _source to write aggregate data to RAM
        _source = MemBucket()
        _features = copy.deepcopy(self.model.features)
        for _, feature in enumerate(self.model.features):
            feature.metric = 'avg'

        i = None
        for i, (_, val, timeval) in enumerate(res):
            bucket = {
                feature.field: val[i]
                for i, feature in enumerate(self.model.features)
            }
            bucket.update({'timestamp': make_ts(timeval)})
            _source.insert_times_data(bucket)

        res2 = _source.get_times_data(
            bucket_interval=self.model.bucket_interval,
            features=self.model.features,
            from_date=self.t0,
            to_date=self.t0 + 9,
        )
        self.model.features = _features

        for i, (_, val2, timeval2) in enumerate(res2):
            (_, val, timeval) = res[i]
            np.testing.assert_allclose(val, val2)

    def test_match_all(self):
        model = Model(dict(
            name="test-model",
            offset=30,
            span=300,
            bucket_interval=3,
            interval=60,
            features=FEATURES_MATCH_ALL_TAG1,
        ))
        res = self.source.get_times_data(
            bucket_interval=model.bucket_interval,
            features=model.features,
            from_date=self.t0,
            to_date=self.t0 + 9,
        )
        baz_avg = []
        for line in res:
            baz_avg.append(nan_to_none(line[1][0]))

        self.assertEqual(
            baz_avg, [216.0, None, 18.0])

        model = Model(dict(
            name="test-model",
            offset=30,
            span=300,
            bucket_interval=3,
            interval=60,
            features=FEATURES_MATCH_ALL_TAG2,
        ))

        res = self.source.get_times_data(
            bucket_interval=model.bucket_interval,
            features=model.features,
            from_date=self.t0,
            to_date=self.t0 + 9,
        )
        baz_avg = []
        for line in res:
            baz_avg.append(nan_to_none(line[1][0]))

        self.assertEqual(
            baz_avg, [-216.0, None, -18.0])

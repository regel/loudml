import datetime
import logging
import os
import unittest

logging.getLogger('tensorflow').disabled = True

from loudml import (
    errors,
)

from loudml.misc import (
    list_from_np,
    ts_to_str,
)

from loudml.timeseries import (
    TimeSeriesModel,
)
from loudml.warp10 import (
    Warp10DataSource,
)

class TestWarp10(unittest.TestCase):
    def setUp(self):
        self.source = Warp10DataSource({
            'name': 'test',
            'url': os.environ.get('url', 'http://localhost:8080/api/v0'),
            'read_token': os.environ['WARP10_READ_TOKEN'],
            'write_token': os.environ['WARP10_WRITE_TOKEN'],
        })

        self.tag = {'test': str(datetime.datetime.now().timestamp())}

        self.model = TimeSeriesModel(dict(
            name="test-model",
            offset=30,
            span=300,
            bucket_interval=3600,
            interval=60,
            features=[
                {
                    'name': 'avg_foo',
                    'metric': 'avg',
                    'measurement': 'measure1',
                    'field': 'foo',
                },
                {
                    'name': 'count_bar',
                    'metric': 'count',
                    'measurement': 'measure2',
                    'field': 'bar',
                    'default': 0,
                },
            ],
            threshold=30,
        ))

    def tearDown(self):
        self.source.drop(tags=self.tag)

    def test_multi_fetch(self):
        res = self.source.build_multi_fetch(
            self.model,
            "2018-07-21T00:00:00Z",
            "2018-07-22T00:00:00Z",
            tags={'key': 'value'},
        )
        self.assertEqual(
            res,
"""
[
[
[
'{}'
'foo'
{{ 'key' 'value' }}
'2018-07-21T00:00:00Z'
'2018-07-22T00:00:00Z'
]
FETCH
bucketizer.mean
0
3600000000
0
]
BUCKETIZE
[
[
'{}'
'bar'
{{ 'key' 'value' }}
'2018-07-21T00:00:00Z'
'2018-07-22T00:00:00Z'
]
FETCH
bucketizer.count
0
3600000000
0
]
BUCKETIZE
]
""".strip().format(self.source.read_token, self.source.read_token)
        )

    def test_write_read(self):
        t0 = datetime.datetime.now(datetime.timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ).timestamp()

        self.source.insert_times_data(
            ts=t0 - 7000,
            data={'foo': 0.7},
            tags=self.tag,
        )
        self.source.insert_times_data(
            ts=t0 - 3800,
            data={'bar': 42},
            tags=self.tag,
        )
        self.source.insert_times_data(
            ts=t0 - 1400,
            data={
                'foo': 0.8,
                'bar': 33,
            },
            tags=self.tag,
        )
        self.source.insert_times_data(
            ts=t0 - 1200,
            data={
                'foo': 0.4,
                'bar': 64,
            },
            tags=self.tag,
        )
        self.source.commit()

        period_len = 3 * self.model.bucket_interval

        res = self.source.get_times_data(
            self.model,
            t0 - period_len,
            t0,
            tags=self.tag,
        )

        self.assertEqual(len(res), period_len / self.model.bucket_interval)

        bucket = res[0][1]
        self.assertEqual(list_from_np(bucket), [None, None])
        bucket = res[1][1]
        self.assertEqual(bucket, [0.7, 1.0])
        bucket = res[2][1]
        self.assertAlmostEqual(bucket[0], 0.6)
        self.assertEqual(bucket[1], 2.0)

    def test_no_data(self):
        with self.assertRaises(errors.NoData):
            self.source.get_times_data(
                self.model,
                "2017-01-01T00:00:00Z",
                "2017-01-31T00:00:00Z",
            )

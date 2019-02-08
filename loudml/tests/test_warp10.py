import loudml.vendor

import datetime
import logging
import os
import random
import numpy as np
import time
import unittest

logging.getLogger('tensorflow').disabled = True

from loudml import (
    errors,
)

from loudml.misc import (
    list_from_np,
    ts_to_str,
)

from loudml.randevents import SinEventGenerator

from loudml.model import Model
from loudml.donut import (
    DonutModel,
)
from loudml.warp10 import (
    Warp10DataSource,
)


class TestWarp10(unittest.TestCase):
    def setUp(self):
        self.prefix = "test-{}".format(datetime.datetime.now().timestamp())
        self.source = Warp10DataSource({
            'name': 'test',
            'url': os.environ['WARP10_URL'],
            'read_token': os.environ['WARP10_READ_TOKEN'],
            'write_token': os.environ['WARP10_WRITE_TOKEN'],
            'global_prefix': self.prefix,
        })
        logger = logging.getLogger('warp10client.client')
        logger.setLevel(logging.INFO)

        self.tag = {'test': self.prefix}

        self.model = Model(dict(
            name="test-model",
            offset=30,
            span=3,
            bucket_interval=3600,
            interval=60,
            features=[
                {
                    'name': 'avg_foo',
                    'metric': 'avg',
                    'field': 'foo',
                },
                {
                    'name': 'count_bar',
                    'metric': 'count',
                    'field': 'bar',
                    'default': 0,
                },
            ],
        ))

        self.t0 = datetime.datetime.now(datetime.timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ).timestamp()

    def tearDown(self):
        self.source.drop(tags=self.tag)

    def test_multi_fetch(self):
        model = Model(dict(
            name="test-model",
            offset=30,
            span=3,
            bucket_interval=3600,
            interval=60,
            features=[
                {
                    'name': 'avg_foo',
                    'metric': 'avg',
                    'field': 'foo',
                    'match_all': [
                        {'tag': 'a', 'value': 'b'},
                    ],
                },
                {
                    'name': 'count_bar',
                    'metric': 'count',
                    'field': 'bar',
                    'default': 0,
                },
            ],
        ))
        res = self.source.build_multi_fetch(
            model,
            "2018-07-21T00:00:00Z",
            "2018-07-22T00:00:00Z",
        )
        self.assertEqual(
            res,
"""
[
[
[
'{}'
'{}.foo'
{{ 'a' 'b' }}
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
'{}.bar'
{{  }}
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
""".strip().format(
               self.source.read_token,
               self.prefix,
               self.source.read_token,
               self.prefix,
            )
        )


    def test_write_read(self):
        t0 = self.t0

        self.source.insert_times_data(
            ts=t0 - 4 * self.model.bucket_interval,
            data={'foo': 0.5},
            tags=self.tag,
        )
        self.source.commit()

        res = self.source.get_times_data(
            self.model,
            t0 - 5 * self.model.bucket_interval,
            t0 - 4 * self.model.bucket_interval,
            tags=self.tag,
        )

        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][1][0], 0.5)
        self.assertEqual(res[0][2], t0 - 5 * self.model.bucket_interval)

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

        _, bucket, ts = res[0]
        self.assertEqual(ts, t0 - period_len)
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

    def test_match_all(self):
        model = Model(dict(
            name="test-model",
            offset=30,
            span=3,
            bucket_interval=3,
            interval=60,
            features=[
                {
                    'name': 'avg_foo',
                    'metric': 'avg',
                    'field': 'foo',
                    'match_all': [
                        {'tag': 'tag_1', 'value': 'tag_A'},
                    ],
                },
            ],
        ))

        t0 = self.t0

        data = [
            # (foo, timestamp)
            (33, t0 - 1), # excluded
            # empty
            (120, t0 + 1), (312, t0 + 2),
            (18, t0 + 7),
            (78, t0 + 10), # excluded
        ]
        for foo, ts in data:
            self.source.insert_times_data(
                ts=ts,
                data={
                    'foo': foo,
                }
            )
            self.source.insert_times_data(
                ts=ts,
                tags={
                    'tag_1': 'tag_A',
                    'tag_2': 'tag_B',
                },
                data={
                    'foo': foo,
                }
            )
            self.source.insert_times_data(
                ts=ts,
                tags={
                    'tag_1': 'tag_B',
                    'tag_2': 'tag_C',
                },
                data={
                    'foo': -foo,
                }
            )

        self.source.commit()
        time.sleep(5)

        res = self.source.get_times_data(
            model,
            from_date=t0,
            to_date=t0 + 3 * model.bucket_interval,
        )
        foo_avg = []
        for line in res:
            foo_avg.append(line[1][0])

        np.testing.assert_allclose(
            np.array(foo_avg),
            np.array([216.0, np.nan, 18.0]),
            rtol=0,
            atol=0,
        )

        model = Model(dict(
            name="test-model",
            offset=30,
            span=3,
            bucket_interval=3,
            interval=60,
            features=[
                {
                    'name': 'avg_foo',
                    'metric': 'avg',
                    'field': 'foo',
                    'match_all': [
                        {'tag': 'tag_1', 'value': 'tag_B'},
                    ],
                },
            ],
        ))

        res = self.source.get_times_data(
            model,
            from_date=self.t0,
            to_date=t0 + 3 * model.bucket_interval,
        )
        avg_foo = []
        for line in res:
            avg_foo.append(line[1][0])

        np.testing.assert_allclose(
            np.array(avg_foo),
            np.array([-216.0, np.nan, -18.0]),
            rtol=0,
            atol=0,
        )

    def test_train_predict(self):
        model = DonutModel(dict(
            name='test',
            offset=30,
            span=5,
            bucket_interval=60 * 60,
            interval=60,
            features=[
                {
                    'name': 'count_foo',
                    'metric': 'count',
                    'field': 'prefix.foo',
                    'default': 0,
                },
                {
                    'name': 'avg_foo',
                    'metric': 'avg',
                    'field': 'prefix.foo',
                    'default': 5,
                },
            ],
            max_evals=1,
        ))

        generator = SinEventGenerator(base=3, sigma=0.05)

        to_date = datetime.datetime.now(datetime.timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ).timestamp()
        from_date = to_date - 3600 * 24

        for ts in generator.generate_ts(from_date, to_date, step_ms=60000):
            self.source.insert_times_data(
                ts=ts,
                data={
                    'prefix.foo': random.lognormvariate(10, 1)
                },
            )

        self.source.commit()

        # Train
        model.train(self.source, from_date=from_date, to_date=to_date)

        # Check
        self.assertTrue(model.is_trained)

        # Predict
        pred_from = to_date - 3 * model.bucket_interval
        pred_to = to_date
        prediction = model.predict(
            datasource=self.source,
            from_date=pred_from,
            to_date=pred_to,
        )
        self.source.save_timeseries_prediction(prediction, model, tags=self.tag)

        # Fake model just for extracting saved prediction
        model2 = Model(dict(
            name='test-prediction',
            offset=30,
            span=5,
            bucket_interval=60 * 60,
            interval=60,
            features=[
                {
                    'name': 'count_foo',
                    'metric': 'avg',
                    'field': "prediction.{}.count_foo".format(model.name),
                },
                {
                    'name': 'avg_foo',
                    'metric': 'avg',
                    'field': "prediction.{}.avg_foo".format(model.name),
                },
            ],
            max_evals=1,
        ))

        res = self.source.get_times_data(
            model2,
            pred_from ,
            pred_to,
            tags=self.tag,
        )

        for i, pred_ts in enumerate(prediction.timestamps):
            values, ts = res[i][1:]
            self.assertEqual(ts, pred_ts)
            np.testing.assert_allclose(
                np.array(values),
                prediction.predicted[i],
            )

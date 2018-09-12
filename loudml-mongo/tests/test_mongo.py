import datetime
import logging
import numpy as np
import os
import random
import unittest

logging.getLogger('tensorflow').disabled = True

from loudml import (
    errors,
)

from loudml.misc import (
    ts_to_str,
)

from loudml.randevents import SinEventGenerator

from loudml.timeseries import (
    TimeSeriesModel,
)
from loudml.mongo import (
    MongoDataSource,
)

class TestMongo(unittest.TestCase):
    def setUp(self):
        db = "test-{}".format(int(datetime.datetime.now().timestamp()))

        self.source = MongoDataSource({
            'name': 'test',
            'addr': os.environ.get('MONGODB_ADDR', "localhost:27017"),
            'db': os.environ.get('MONODB_DB', db),
        })

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
                    'collection': 'coll',
                    'field': 'foo',
                },
                {
                    'name': 'count_bar',
                    'metric': 'count',
                    'collection': 'coll',
                    'field': 'bar',
                    'default': 0,
                },
            ],
            threshold=30,
        ))

        self.t0 = datetime.datetime.now(datetime.timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ).timestamp()

    def tearDown(self):
        self.source.drop()

    def test_write_read(self):
        t0 = self.t0

        self.source.insert_times_data(
            collection='coll',
            ts=t0 - 7000,
            data={'foo': 0.7},
        )
        self.source.insert_times_data(
            ts=t0 - 3800,
            collection='coll',
            data={'bar': 42},
        )
        self.source.insert_times_data(
            ts=t0 - 1400,
            collection='coll',
            data={
                'foo': 0.8,
                'bar': 33,
            },
        )
        self.source.insert_times_data(
            ts=t0 - 1200,
            collection='coll',
            data={
                'foo': 0.4,
                'bar': 64,
            },
        )
        self.source.commit()

        res = self.source.get_times_data(
            self.model,
            t0 - 7200,
            t0,
        )

        bucket = res[0][1]
        self.assertEqual(bucket, [0.7, 1.0])
        bucket = res[1][1]
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
        model = TimeSeriesModel(dict(
            name="test-model",
            offset=30,
            span=300,
            bucket_interval=3,
            interval=60,
            features=[
                {
                    'name': 'avg_foo',
                    'metric': 'avg',
                    'collection': 'coll1',
                    'field': 'foo',
                    'match_all': [
                        {'tag': 'tag_1', 'value': 'tag_A'},
                    ],
                },
            ],
            threshold=30,
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
                collection='coll1',
                ts=ts,
                data={
                    'foo': foo,
                }
            )
            self.source.insert_times_data(
                collection='coll1',
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
                collection='coll1',
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
        model = TimeSeriesModel(dict(
            name="test-model",
            offset=30,
            span=300,
            bucket_interval=3,
            interval=60,
            features=[
                {
                    'collection': 'coll1',
                    'name': 'avg_foo',
                    'metric': 'avg',
                    'field': 'foo',
                    'match_all': [
                        {'tag': 'tag_1', 'value': 'tag_B'},
                    ],
                },
            ],
            threshold=30,
        ))
        res = self.source.get_times_data(
            model,
            from_date=self.t0,
            to_date=self.t0 + 8,
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
        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=5,
            bucket_interval=60 * 60,
            interval=60,
            features=[
                {
                    'name': 'count_foo',
                    'metric': 'count',
                    'collection': 'coll',
                    'field': 'foo',
                    'default': 0,
                },
                {
                    'name': 'avg_foo',
                    'metric': 'avg',
                    'collection': 'coll',
                    'field': 'foo',
                    'default': 5,
                },
            ],
            threshold=30,
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
                collection="coll",
                ts=ts,
                data={
                    'foo': random.lognormvariate(10, 1)
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
        self.source.save_timeseries_prediction(prediction, model)

        boundaries = list(range(
            int(pred_from),
            int(pred_to + model.bucket_interval),
            int(model.bucket_interval),
        ))

        res = self.source.db['prediction_test'].aggregate([
            {'$bucket': {
                'groupBy': '$timestamp',
                'boundaries': boundaries,
                'default': None,
                'output': {
                    'count_foo': {'$avg': '$count_foo'},
                    'avg_foo': {'$avg': '$avg_foo'},
                }
            }}
        ])
        pred_buckets = prediction.format_buckets()
        for i, entry in enumerate(res):
            predicted = pred_buckets[i]['predicted']
            self.assertEqual(predicted['count_foo'], entry['count_foo'])
            self.assertEqual(predicted['avg_foo'], entry['avg_foo'])

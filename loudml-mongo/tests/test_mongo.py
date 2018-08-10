import datetime
import logging
import os
import random
import unittest

logging.getLogger('tensorflow').disabled = True

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

    def tearDown(self):
        self.source.drop()

    def test_write_read(self):
        t0 = datetime.datetime.now(datetime.timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ).timestamp()

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

    def test_train(self):
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

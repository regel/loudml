from loudml.mongo import (
    MongoBucket,
)
from loudml.donut import (
    DonutModel,
)
from loudml.model import Model
from randevents import SinEventGenerator
from loudml import (
    errors,
)

import datetime
import logging
import numpy as np
import os
import random
import unittest

logging.getLogger('tensorflow').disabled = True


class TestMongo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db = os.environ.get('MONGODB_DB')
        if not db:
            db = "test-{}".format(int(datetime.datetime.now().timestamp()))

        settings = {
            'name': 'test',
            'addr': os.environ.get('MONGODB_ADDR', "localhost:27017"),
            'database': db,
            'collection': 'coll',
        }

        username = os.environ.get('MONGODB_USER')

        if username:
            settings['username'] = username
            settings['password'] = os.environ.get('MONGODB_PWD')

            auth_source = os.environ.get('MONGODB_AUTH_SRC')
            if auth_source:
                settings['auth_source'] = auth_source

        cls.bucket_cfg = settings
        cls.source = MongoBucket(settings)

        cls.model = Model(dict(
            name="test-model",
            offset=30,
            span=300,
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

        cls.t0 = datetime.datetime.now(datetime.timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ).timestamp()

    @classmethod
    def tearDownClass(cls):
        cls.source.drop()

    def test_validate_config(self):
        MongoBucket({
            'name': 'test',
            'addr': "localhost:27017",
            'database': "mydb",
            'username': 'obelix',
            'password': 'sangliers',
            'collection': 'coll',
        })

    def test_write_read(self):
        t0 = self.t0

        self.source.insert_times_data(
            ts=t0 - 7000,
            data={'foo': 0.7},
        )
        self.source.insert_times_data(
            ts=t0 - 3800,
            data={'bar': 42},
        )
        self.source.insert_times_data(
            ts=t0 - 1400,
            data={
                'foo': 0.8,
                'bar': 33,
            },
        )
        self.source.insert_times_data(
            ts=t0 - 1200,
            data={
                'foo': 0.4,
                'bar': 64,
            },
        )
        self.source.commit()

        res = self.source.get_times_data(
            bucket_interval=self.model.bucket_interval,
            features=self.model.features,
            from_date=t0 - 7200,
            to_date=t0,
        )

        bucket = res[0][1]
        self.assertEqual(bucket, [0.7, 1.0])
        bucket = res[1][1]
        self.assertAlmostEqual(bucket[0], 0.6)
        self.assertEqual(bucket[1], 2.0)

    def test_no_data(self):
        with self.assertRaises(errors.NoData):
            self.source.get_times_data(
                bucket_interval=self.model.bucket_interval,
                features=self.model.features,
                from_date="2017-01-01T00:00:00Z",
                to_date="2017-01-31T00:00:00Z",
            )

    def test_match_all(self):
        settings = self.bucket_cfg
        settings['collection'] = 'coll1'
        source = MongoBucket(settings)

        model = Model(dict(
            name="test-model",
            offset=30,
            span=300,
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
            (33, t0 - 1),  # excluded
            # empty
            (120, t0 + 1), (312, t0 + 2),
            (18, t0 + 7),
            (78, t0 + 10),  # excluded
        ]
        for foo, ts in data:
            source.insert_times_data(
                ts=ts,
                data={
                    'foo': foo,
                }
            )
            source.insert_times_data(
                ts=ts,
                tags={
                    'tag_1': 'tag_A',
                    'tag_2': 'tag_B',
                },
                data={
                    'foo': foo,
                }
            )
            source.insert_times_data(
                ts=ts,
                tags={
                    'tag_1': 'tag_B',
                    'tag_2': 'tag_C',
                },
                data={
                    'foo': -foo,
                }
            )
        source.commit()
        res = source.get_times_data(
            bucket_interval=model.bucket_interval,
            features=model.features,
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
            span=300,
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
        res = source.get_times_data(
            bucket_interval=model.bucket_interval,
            features=model.features,
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

    @unittest.skip("no mv support yet in donut model")
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
                    'field': 'foo',
                    'default': 0,
                },
                {
                    'name': 'avg_foo',
                    'metric': 'avg',
                    'field': 'foo',
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
            bucket=self.source,
            from_date=pred_from,
            to_date=pred_to,
        )
        self.source.save_timeseries_prediction(
            prediction, tags=model.get_tags())

        boundaries = list(range(
            int(pred_from),
            int(pred_to + model.bucket_interval),
            int(model.bucket_interval),
        ))

        res = self.source.db['loudml'].aggregate([
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

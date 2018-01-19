import datetime
import logging
import math
import os
import random
import unittest

logging.getLogger('tensorflow').disabled = True

from loudml.randevents import SinEventGenerator
from loudml.times import TimesModel
from loudml.memdatasource import MemDataSource
from loudml.memstorage import MemStorage

FEATURES = [
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
        'default': 10,
    },
]

class TestTimes(unittest.TestCase):
    def setUp(self):
        self.source = MemDataSource()
        self.storage = MemStorage()
        self.model = None

        generator = SinEventGenerator(avg=3, sigma=0.05)

        self.to_date = datetime.datetime.now().timestamp()
        self.from_date = self.to_date - 3600 * 24 * 7

        for ts in generator.generate_ts(self.from_date, self.to_date, step=600):
            self.source.insert_times_data({
                'timestamp': ts,
                'foo': random.normalvariate(10, 1)
            })

    def _require_training(self):
        if self.model:
            return

        self.model = TimesModel(dict(
            name='test',
            offset=30,
            span=5,
            bucket_interval=20 * 60,
            interval=60,
            features=FEATURES,
            threshold=30,
            max_evals=10,
        ))

        self.model.train(self.source, self.from_date, self.to_date)

    def test_train(self):
        self._require_training()
        self.assertTrue(self.model.is_trained)

    def test_format(self):
        import numpy as np

        data = [0, 2, 4, 6, 8, 10, 12, 14]
        dataset = np.zeros((8, 1), dtype=float)
        for i, val in enumerate(data):
            dataset[i] = val

        model = TimesModel(dict(
            name='test_fmt',
            offset=30,
            span=3,
            bucket_interval=20 * 60,
            interval=60,
            features=FEATURES,
            threshold=30,
            max_evals=10,
        ))

        indexes, x, y = model._format_dataset(dataset)

        self.assertEqual(indexes.tolist(), [3, 4, 5, 6, 7])
        self.assertEqual(x.tolist(), [
            [[0], [2], [4]],
            [[2], [4], [6]],
            [[4], [6], [8]],
            [[6], [8], [10]],
            [[8], [10], [12]],
        ])
        self.assertEqual(y.tolist(), [[6], [8], [10], [12], [14]])

    def test_train(self):
        self._require_training()
        self.assertTrue(self.model.is_trained)

    def test_predict(self):
        to_date = self.to_date
        from_date = to_date - 24 * 3600

        self._require_training()
        prediction = self.model.predict(self.source, from_date, to_date)

        expected = math.ceil(
            (to_date - from_date) / self.model.bucket_interval
        )

        obs_avg = prediction.observed['avg_foo']
        obs_count = prediction.observed['count_foo']
        pred_avg = prediction.predicted['avg_foo']
        pred_count = prediction.predicted['count_foo']

        self.assertEqual(len(prediction.timestamps), expected)
        self.assertEqual(len(obs_avg), expected)
        self.assertEqual(len(pred_avg), expected)
        self.assertEqual(len(obs_count), expected)
        self.assertEqual(len(pred_count), expected)

        for i in range(expected):
            self.assertAlmostEqual(
                pred_avg[i], obs_avg[i], delta=2.0,
            )
            self.assertAlmostEqual(
                pred_count[i], obs_count[i], delta=12,
            )

    def test_predict_with_nan(self):
        source = MemDataSource()
        storage = MemStorage()

        to_date = datetime.datetime.now().timestamp()
        from_date = to_date - 3600 * 24 * 7 * 3

        # Generate 3 weeks of data
        ts = from_date
        for i in range(7 * 3):
            # Generate 23h of data
            for j in range(23):
                source.insert_times_data({
                    'timestamp': ts,
                    'foo': j,
                })
                ts += 3600

            # No data for last hour
            ts += 3600

        model = TimesModel(dict(
            name='test',
            offset=30,
            span=3,
            bucket_interval=3600,
            interval=60,
            features=[{
                'name': 'avg_foo',
                'metric': 'avg',
                'field': 'foo',
                #'default': None,
            }],
            threshold=30,
            max_evals=10,
        ))

        model.train(source, from_date, to_date)
        self.assertTrue(model.is_trained)

        from_date = to_date - 3600 * 24
        prediction = model.predict(source, from_date, to_date)

        obs = prediction.observed['avg_foo']
        pred = prediction.predicted['avg_foo']

        self.assertEqual(len(prediction.timestamps), 24)
        self.assertEqual(len(obs), 24)
        self.assertEqual(len(pred), 24)

        # Holes expected in prediction
        self.assertEqual(pred[:model.span], [None] * model.span)
        self.assertEqual(pred[-1], None)

        for i in range(24):
            self.assertAlmostEqual(
                pred[i], obs[i], delta=2.0,
            )

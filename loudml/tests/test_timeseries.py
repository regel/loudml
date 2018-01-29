import datetime
import logging
import math
import os
import random
import unittest

logging.getLogger('tensorflow').disabled = True

from loudml.randevents import SinEventGenerator
from loudml.timeseries import TimeSeriesModel
from loudml.memdatasource import MemDataSource
from loudml.memstorage import MemStorage

from loudml import (
    errors,
)

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
        self.from_date = self.to_date - 3600 * 24 * 7 * 2

        for ts in generator.generate_ts(self.from_date, self.to_date, step=600):
            self.source.insert_times_data({
                'timestamp': ts,
                'foo': random.normalvariate(10, 1)
            })

    def test_validation(self):
        valid = {
            'name': 'foo',
            'bucket_interval': '20m',
            'interval': '10m',
            'offset': 10,
            'span': 3,
            'features': [
                {
                    'name': 'avg_foo',
                    'metric': 'avg',
                    'field': 'foo',
                }
            ]
        }

        model = TimeSeriesModel(valid)
        self.assertEqual(model.bucket_interval, 20 * 60)
        self.assertEqual(model.interval, 10 * 60)
        self.assertEqual(model.offset, 10)
        self.assertEqual(model.span, 3)
        self.assertEqual(len(model.features), 1)

        def invalid(key, value):
            settings = valid.copy()
            settings[key] = value

            with self.assertRaises(errors.Invalid):
                TimeSeriesModel(settings)

        invalid('bucket_interval', 0)
        invalid('interval', 0)
        invalid('offset', -1)
        invalid('span', 0)

    def _require_training(self):
        if self.model:
            return

        self.model = TimeSeriesModel(dict(
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

        model = TimeSeriesModel(dict(
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

    def test_predict_aligned(self):
        self._require_training()

        to_date = math.floor(self.to_date / self.model.bucket_interval) * self.model.bucket_interval
        from_date = to_date - 24 * 3600

        prediction = self.model.predict(self.source, from_date, to_date)

        expected = math.ceil(
            (to_date - from_date) / self.model.bucket_interval
        )

        # prediction.plot('count_foo')

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

    def test_predict_unaligned(self):
        self._require_training()

        # Aligned
        to_date = math.floor(self.to_date / self.model.bucket_interval) * self.model.bucket_interval
        # Unaligned
        to_date += self.model.bucket_interval / 4
        from_date = to_date

        prediction = self.model.predict(self.source, from_date, to_date)

        expected = math.ceil(
            (to_date - from_date) / self.model.bucket_interval
        )

        obs_avg = prediction.observed['avg_foo']
        obs_count = prediction.observed['count_foo']
        pred_avg = prediction.predicted['avg_foo']
        pred_count = prediction.predicted['count_foo']

        self.assertEqual(len(prediction.timestamps), 1)
        self.assertEqual(len(obs_avg), 1)
        self.assertEqual(len(pred_avg), 1)
        self.assertEqual(len(obs_count), 1)
        self.assertEqual(len(pred_count), 1)

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

        to_date = datetime.datetime.now().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ).timestamp()

        # Generate 3 days of data
        nb_days = 3
        hist_to = to_date
        hist_from = to_date - 3600 * 24 * nb_days
        ts = hist_from

        for i in range(nb_days):
            # [0h-12h[
            for j in range(12):
                source.insert_times_data({
                    'timestamp': ts,
                    'foo': j,
                })
                ts += 3600

            # No data for [12h, 13h[
            ts += 3600

            # [13h-0h[
            for j in range(11):
                source.insert_times_data({
                    'timestamp': ts,
                    'foo': j,
                })
                ts += 3600

        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=3,
            bucket_interval=3600,
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
                   'default': None,
                },
            ],
            threshold=30,
            max_evals=1,
        ))

        # train on all dataset
        model.train(source, hist_from, hist_to)
        self.assertTrue(model.is_trained)

        # predict on last 24h
        to_date = hist_to
        from_date = to_date - 3600 * 24
        prediction = model.predict(source, from_date, to_date)

        obs = prediction.observed['avg_foo']
        pred = prediction.predicted['avg_foo']

        self.assertEqual(len(prediction.timestamps), 24)
        self.assertEqual(len(obs), 24)
        self.assertEqual(len(pred), 24)

        # Holes expected in prediction
        self.assertEqual(pred[13:13+model.span], [None] * model.span)

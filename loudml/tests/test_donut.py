import loudml.vendor

import datetime
import logging
import math
import os
import random
import unittest

import numpy as np

def nan_equal(a,b):
    try:
        np.testing.assert_equal(a,b)
    except AssertionError:
        return False
    return True

logging.getLogger('tensorflow').disabled = True

import loudml.vendor

from loudml.randevents import (
    FlatEventGenerator,
    SinEventGenerator,
    TriangleEventGenerator,
)
from loudml.donut import (
    DonutModel,
    TimeSeriesPrediction,
)
from loudml.model import Feature

from loudml.memdatasource import MemDataSource
from loudml.filestorage import TempStorage
from loudml.misc import (
    make_datetime,
    make_ts,
    dt_get_daytime,
    dt_get_weekday,
    ts_to_str,
)

from loudml import (
    errors,
)

from loudml.api import Hook

FEATURE_COUNT_FOO = {
    'name': 'count_foo',
    'metric': 'count',
    'field': 'foo',
    'default': 0,
}

FEATURE_AVG_FOO = {
    'name': 'avg_foo',
    'metric': 'avg',
    'field': 'foo',
    'default': 10,
}

FEATURES = [FEATURE_COUNT_FOO]

class TestHook(Hook):
    def __init__(self, model, storage, *args, **kwargs):
        super().__init__(
            "test",
            None,
            model,
            storage,
            *args,
            **kwargs
        )
        self.events = []

    def on_anomaly_start(
        self,
        dt,
        score,
        predicted,
        observed,
        anomalies,
        *args,
        **kwargs
    ):
        for feature_name, ano in anomalies.items():
            logging.error(
                "feature '{}' is too {} (score = {:.1f})".format(
                     self.feature_to_str(feature_name),
                     ano['type'],
                     ano['score']
                )
            )

        self.events.append({
            'type': 'start',
            'dt': dt,
        })

    def on_anomaly_end(self, dt, *args, **kwargs):
        self.events.append({
            'type': 'end',
            'dt': dt,
        })


class TestTimes(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.source = MemDataSource()
        self.storage = TempStorage()

        self.model = DonutModel(dict(
            name='test',
            offset=30,
            span=24 * 3,
            bucket_interval=20 * 60,
            interval=60,
            features=FEATURES,
            grace_period="140m", # = 7 points
            max_threshold=99.7,
            min_threshold=68,
            max_evals=10,
        ))

        self.generator = SinEventGenerator(base=3, amplitude=3, sigma=0.01)

        to_date = datetime.datetime.now().timestamp()

        # Be sure that date range is aligned
        self.to_date = math.floor(to_date / self.model.bucket_interval) * self.model.bucket_interval
        self.from_date = self.to_date - 3600 * 24 * 7 * 3

        for ts in self.generator.generate_ts(self.from_date, self.to_date, step_ms=600000):
            self.source.insert_times_data({
                'timestamp': ts,
                'foo': random.normalvariate(10, 1)
            })


    def _require_training(self):
        if self.model.is_trained:
            return

        self.model.train(self.source, self.from_date, self.to_date)

    def test_train(self):
        self._require_training()
        self.assertTrue(self.model.is_trained)


    def test_format(self):
        dataset = np.array([0, np.nan, 4, 6, 8, 10, 12, 14])
        model = DonutModel(dict(
            name='test_fmt',
            offset=30,
            span=3,
            bucket_interval=20 * 60,
            interval=60,
            features=[
                FEATURE_COUNT_FOO,
            ],
            max_evals=1,
        ))

        missing, x = model._format_dataset(dataset)
        self.assertEqual(missing.tolist(), [
            [False, True, False],
            [True, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ])
        self.assertEqual(x.tolist(), [
            [0.0, 0.0, 4.0],
            [0.0, 4.0, 6.0],
            [4.0, 6.0, 8.0],
            [6.0, 8.0, 10.0],
            [8.0, 10.0, 12.0],
            [10.0, 12.0, 14.0],
        ])

    def test_train(self):
        self._require_training()
        self.assertTrue(self.model.is_trained)

    def test_forecast(self):
        model = DonutModel(dict(
            name='test',
            offset=30,
            span=100,
            forecast=1,
            bucket_interval=20 * 60,
            interval=60,
            features=[
                FEATURE_COUNT_FOO,
            ],
            max_evals=5,
        ))
        source = MemDataSource()
        generator = SinEventGenerator(base=3, amplitude=3, sigma=0.01)

        # Align date range to day interval
        to_date = datetime.datetime.now().timestamp()
        to_date = math.floor(to_date / (3600*24)) * (3600*24)
        from_date = to_date - 3600 * 24 * 7 * 3
        for ts in generator.generate_ts(from_date, to_date, step_ms=600000):
            source.insert_times_data({
                'timestamp': ts,
                'foo': random.normalvariate(10, 1)
            })

        model.train(source, from_date, to_date)
        prediction = model.predict(source, from_date, to_date)

        from_date = to_date
        to_date = from_date + 48 * 3600
        forecast = model.forecast(source, from_date, to_date)

        expected = math.ceil(
            (to_date - from_date) / model.bucket_interval
        )

        self.assertEqual(len(forecast.timestamps), expected)
        self.assertEqual(forecast.observed.shape, (expected,))
        self.assertEqual(forecast.predicted.shape, (expected,))

        all_default = np.full(
            (expected,),
            model.features[0].default,
            dtype=float,
        )
        np.testing.assert_allclose(
            forecast.observed,
            all_default,
        )

        forecast_head = np.array([7.52, 7.88, 8.61, 9.06, 9.54])
        forecast_tail = np.array([5.02, 5.46, 6.11, 6.74, 7.16])

#        print(forecast.predicted)
        delta = 1.0
        forecast_good = np.abs(forecast.predicted[:len(forecast_head)] - forecast_head) <= delta
#        print(forecast_head)
#        print(forecast.predicted[:len(forecast_head)])
#        print(forecast_good)
        self.assertEqual(np.all(forecast_good), True)
        forecast_good = np.abs(forecast.predicted[-len(forecast_tail):] - forecast_tail) <= delta
#        print(forecast_tail)
#        print(forecast.predicted[-len(forecast_tail):])
#        print(forecast_good)
        self.assertEqual(np.all(forecast_good), True)

    def test_predict_aligned(self):
        self._require_training()

        to_date = self.to_date
        from_date = to_date - 24 * 3600

        prediction = self.model.predict(self.source, from_date, to_date)

        expected = math.ceil(
            (to_date - from_date) / self.model.bucket_interval
        )

        # prediction.plot('count_foo')

        self.assertEqual(len(prediction.timestamps), expected)
        self.assertEqual(prediction.observed.shape, (expected,))
        self.assertEqual(prediction.predicted.shape, (expected,))

        for i in range(expected):
            self.assertAlmostEqual(
                prediction.observed[i],
                prediction.predicted[i],
                delta=2,
            )

    def test_predict_with_nan(self):
        source = MemDataSource()
        storage = TempStorage()

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

        model = DonutModel(dict(
            name='test',
            offset=30,
            span=24,
            bucket_interval=3600,
            interval=60,
            features=[
                {
                   'name': 'count_foo',
                   'metric': 'count',
                   'field': 'foo',
                },
            ],
            max_threshold=30,
            min_threshold=25,
            max_evals=10,
        ))

        # train on all dataset
        model.train(source, hist_from, hist_to)
        self.assertTrue(model.is_trained)

        # predict on last 24h
        to_date = hist_to
        from_date = to_date - 3600 * 24
        prediction = model.predict(source, from_date, to_date)

        # prediction.plot('count_foo')

        self.assertEqual(len(prediction.timestamps), 24)
        self.assertEqual(prediction.observed.shape, (24,))
        self.assertEqual(prediction.predicted.shape, (24,))

        # Adding this call to ensure detect_anomalies() can deal with nan
        model.detect_anomalies(prediction)

        # Donut does missing data insertion and can fill the gap in the data
        for i in range(24):
            self.assertAlmostEqual(
                1.0,
                prediction.predicted[i],
                delta=0.22,
            )


    def test_detect_anomalies(self):
        self._require_training()

        source = MemDataSource()

        bucket_interval = self.model.bucket_interval

        # Insert 1000 buckets of normal data
        to_date = datetime.datetime.now().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ).timestamp()
        from_date = to_date - 1000 * bucket_interval

        for ts in self.generator.generate_ts(from_date, to_date, step_ms=600000):
            source.insert_times_data({
                'timestamp': ts,
                'foo': random.normalvariate(10, 1)
            })

        # Add abnormal data
        generator = FlatEventGenerator(base=5, sigma=0.01)

        from_date = to_date - 20 * bucket_interval
        for i in [5, 6, 7, 17, 18, 19]:
            ano_from = from_date + i * bucket_interval
            ano_to = ano_from + 1 * bucket_interval
            for ts in generator.generate_ts(ano_from, ano_to, step_ms=600000):
                source.insert_times_data({
                    'timestamp': ts,
                    'foo': random.normalvariate(10, 1)
                })

        # Make prediction on buckets [0-20[
        prediction = self.model.predict2(
            source,
            from_date,
            to_date,
            mse_rtol=0, # unused
        )

        self.model.detect_anomalies(prediction)

        buckets = prediction.format_buckets()

        assert len(buckets) == 20

#        import json
#        print(json.dumps(buckets, indent=4))
#        prediction.plot('count_foo')

        # Buckets [0-4] are normal
        for i in range(0, 5):
            self.assertFalse(buckets[i]['stats']['anomaly'])

        # Bucket 5 is abnormal
        self.assertTrue(buckets[5]['stats']['anomaly'])
        # Bucket 6 is abnormal
        self.assertTrue(buckets[6]['stats']['anomaly'])
        # Bucket 7 is abnormal
        self.assertTrue(buckets[7]['stats']['anomaly'])

        # lag: 8 and 9 for cool down time
        # Buckets [8-16] are in grace period and expected to be normal
        for i in range(10, 17):
            self.assertFalse(buckets[i]['stats']['anomaly'])

        # Bucket 17 and 18 and 19 are abnormal
        self.assertTrue(buckets[17]['stats']['anomaly'])
        self.assertTrue(buckets[18]['stats']['anomaly'])
        self.assertTrue(buckets[19]['stats']['anomaly'])

        anomalies = prediction.get_anomalies()
        self.assertEqual(
            anomalies[0:3],
            [buckets[i] for i in [5, 6, 7]],
        )
        self.assertEqual(
            anomalies[-3:],
            [buckets[i] for i in [17, 18, 19]],
        )


    def test_thresholds(self):
        source = MemDataSource()
        to_date = datetime.datetime.now().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=datetime.timezone.utc,
        ).timestamp()

        # Generate 3 weeks days of data
        nb_days = 3 * 7
        hist_to = to_date
        hist_from = to_date - 3600 * 24 * nb_days
        ts = hist_from
        value = 5

        for i in range(nb_days):
            for j in range(0, 24):
                source.insert_times_data({
                    'timestamp': ts,
                    'foo': value,
                })
                ts += 3600

        model = DonutModel(dict(
            name='test',
            offset=30,
            span=24*3,
            bucket_interval=3600,
            interval=60,
            features=[
                {
                   'name': 'avg_foo',
                   'metric': 'avg',
                   'field': 'foo',
                   'default': 0,
                },
            ],
            max_threshold=99.7,
            min_threshold=68,
            max_evals=5,
        ))

        model.train(source, hist_from, hist_to)
        self.assertTrue(model.is_trained)

        # Add an extra day
        ts = hist_to
        values = []

        # Normal value on [00:00-06:00[
        values += [value] * 6

        # Increase on [06:00-12:00[
        values += list(range(value, value + 6))

        # Decrease on [12:00-18:00[
        values += list(range(value + 6, value, -1))

        # Normal value on [18:00-24:00[
        values += [value] * 6

        for value in values:
            source.insert_times_data({
                'timestamp': ts,
                'foo': value,
            })
            ts += 3600

        prediction = model.predict(source, hist_to, ts)
        self.assertEqual(len(prediction.timestamps), 24)

        hook = TestHook(model.settings, self.storage)

        model.detect_anomalies(prediction, hooks=[hook])
        self.assertEqual(len(hook.events), 2)
        event0, event1 = hook.events
        self.assertEqual(event0['type'], 'start')
        self.assertEqual(event1['type'], 'end')
        self.assertGreaterEqual(
            (event1['dt'] - event0['dt']).seconds,
            6 * 3600,
        )


    def test_thresholds2(self):
        source = MemDataSource()
        to_date = datetime.datetime.now().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=datetime.timezone.utc,
        ).timestamp()

        # Generate 3 weeks days of data
        nb_days = 3 * 7
        hist_to = to_date
        hist_from = to_date - 3600 * 24 * nb_days
        ts = hist_from
        value = 5

        for i in range(nb_days):
            for j in range(0, 24):
                source.insert_times_data({
                    'timestamp': ts,
                    'foo': value + random.normalvariate(0, 1),
                })
                ts += 3600

        model = DonutModel(dict(
            name='test',
            offset=30,
            span=24*3,
            bucket_interval=3600,
            interval=60,
            features=[
                {
                   'name': 'avg_foo',
                   'metric': 'avg',
                   'field': 'foo',
                   'default': 0,
                   'anomaly_type': 'low',
                },
            ],
            max_threshold=99.7,
            min_threshold=68,
            max_evals=5,
        ))

        model.train(source, hist_from, hist_to)
        self.assertTrue(model.is_trained)

        # Add an extra day
        ts = hist_to
        values = []

        # Normal value on [00:00-06:00[
        values += [value] * 6

        # Decrease on [06:00-12:00[
        values += list(range(value, value - 6, -1))

        # Increase on [12:00-18:00[
        values += list(range(value - 6, value, 1))

        # Normal value on [18:00-24:00[
        values += [value] * 6

        for value in values:
            source.insert_times_data({
                'timestamp': ts,
                'foo': value,
            })
            ts += 3600

        prediction = model.predict(source, hist_to, ts)
        self.assertEqual(len(prediction.timestamps), 24)

        hook = TestHook(model.settings, self.storage)

        model.detect_anomalies(prediction, hooks=[hook])

        buckets = prediction.format_buckets()
        # 68–95–99.7 rule
        self.assertEqual(buckets[7]['stats']['anomalies']['avg_foo']['type'], 'low')
        self.assertAlmostEqual(buckets[7]['stats']['anomalies']['avg_foo']['score'], 100, delta=35)
        self.assertEqual(buckets[8]['stats']['anomalies']['avg_foo']['type'], 'low')
        self.assertAlmostEqual(buckets[8]['stats']['anomalies']['avg_foo']['score'], 100, delta=5)
        self.assertEqual(buckets[9]['stats']['anomalies']['avg_foo']['type'], 'low')
        self.assertAlmostEqual(buckets[9]['stats']['anomalies']['avg_foo']['score'], 100, delta=2)

        self.assertEqual(len(hook.events), 2)
        event0, event1 = hook.events
        self.assertEqual(event0['type'], 'start')
        self.assertEqual(event1['type'], 'end')
        self.assertGreaterEqual(
            (event1['dt'] - event0['dt']).seconds,
            6 * 3600,
        )

    def test_low(self):
        source = MemDataSource()
        to_date = datetime.datetime.now().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=datetime.timezone.utc,
        ).timestamp()

        # Generate 1 week days of data
        nb_days = 7
        hist_to = to_date
        hist_from = to_date - 3600 * 24 * nb_days
        ts = hist_from

        for i in range(nb_days):
            for j in range(0, 24):
                source.insert_times_data({
                    'timestamp': ts,
                    'foo': random.randrange(45, 55),
#                    'bar': random.randrange(45, 55),
#                    'baz': random.randrange(45, 55),

                })
                ts += 3600

        model = DonutModel(dict(
            name='test',
            offset=30,
            span=24,
            bucket_interval=3600,
            interval=60,
            features=[
                {
                   'name': 'avg_foo',
                   'metric': 'avg',
                   'field': 'foo',
                   'anomaly_type': 'low',
                },
            ],
            max_threshold=99.7,
            min_threshold=65,
            max_evals=5,
        ))

        model.train(source, hist_from, hist_to)
        self.assertTrue(model.is_trained)

        ts = hist_to
        data = [
            [20.0, 50.0, 80.0],
            [50.0, 80.0, 50.0],
            [50.0, 50.0, 20.0],
        ]

        for values in data:
            source.insert_times_data({
                'timestamp': ts,
                'foo': values[0],
#                'bar': values[1],
#                'baz': values[2],
            })
            ts += 3600

        prediction = model.predict(source, hist_to, ts)
        self.assertEqual(len(prediction.timestamps), 3)

        model.detect_anomalies(prediction)

        buckets = prediction.format_buckets()

        anomalies = buckets[0]['stats']['anomalies']
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies['avg_foo']['type'], 'low')

        anomalies = buckets[1]['stats']['anomalies']
        self.assertEqual(len(anomalies), 0)

        anomalies = buckets[2]['stats']['anomalies']
        self.assertEqual(len(anomalies), 0)

    def test_high(self):
        source = MemDataSource()
        to_date = datetime.datetime.now().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=datetime.timezone.utc,
        ).timestamp()

        # Generate 1 week days of data
        nb_days = 7
        hist_to = to_date
        hist_from = to_date - 3600 * 24 * nb_days
        ts = hist_from

        for i in range(nb_days):
            for j in range(0, 24):
                source.insert_times_data({
                    'timestamp': ts,
#                    'foo': random.randrange(45, 55),
                    'bar': random.randrange(45, 55),
#                    'baz': random.randrange(45, 55),

                })
                ts += 3600

        model = DonutModel(dict(
            name='test',
            offset=30,
            span=24,
            bucket_interval=3600,
            interval=60,
            features=[
                {
                   'name': 'avg_bar',
                   'metric': 'avg',
                   'field': 'bar',
                   'anomaly_type': 'high',
                },
            ],
            max_threshold=99.7,
            min_threshold=65,
            max_evals=5,
        ))

        model.train(source, hist_from, hist_to)
        self.assertTrue(model.is_trained)

        ts = hist_to
        data = [
            [20.0, 50.0, 80.0],
            [50.0, 80.0, 50.0],
            [50.0, 50.0, 20.0],
        ]

        for values in data:
            source.insert_times_data({
                'timestamp': ts,
#                'foo': values[0],
                'bar': values[1],
#                'baz': values[2],
            })
            ts += 3600

        prediction = model.predict(source, hist_to, ts)
        self.assertEqual(len(prediction.timestamps), 3)

        model.detect_anomalies(prediction)

        buckets = prediction.format_buckets()

        anomalies = buckets[0]['stats']['anomalies']
        self.assertEqual(len(anomalies), 0)

        anomalies = buckets[1]['stats']['anomalies']
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies['avg_bar']['type'], 'high')

        anomalies = buckets[2]['stats']['anomalies']
        self.assertEqual(len(anomalies), 0)

    def test_low_high(self):
        source = MemDataSource()
        to_date = datetime.datetime.now().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=datetime.timezone.utc,
        ).timestamp()

        # Generate 1 week days of data
        nb_days = 7
        hist_to = to_date
        hist_from = to_date - 3600 * 24 * nb_days
        ts = hist_from

        for i in range(nb_days):
            for j in range(0, 24):
                source.insert_times_data({
                    'timestamp': ts,
#                    'foo': random.randrange(45, 55),
#                    'bar': random.randrange(45, 55),
                    'baz': random.randrange(45, 55),

                })
                ts += 3600

        model = DonutModel(dict(
            name='test',
            offset=30,
            span=24,
            bucket_interval=3600,
            interval=60,
            features=[
                {
                   'name': 'avg_baz',
                   'metric': 'avg',
                   'field': 'baz',
                   'anomaly_type': 'low_high',
                },
            ],
            max_threshold=99.7,
            min_threshold=65,
            max_evals=5,
        ))

        model.train(source, hist_from, hist_to)
        self.assertTrue(model.is_trained)

        ts = hist_to
        data = [
            [20.0, 50.0, 80.0],
            [50.0, 80.0, 50.0],
            [50.0, 50.0, 20.0],
        ]

        for values in data:
            source.insert_times_data({
                'timestamp': ts,
#                'foo': values[0],
#                'bar': values[1],
                'baz': values[2],
            })
            ts += 3600

        prediction = model.predict(source, hist_to, ts)
        self.assertEqual(len(prediction.timestamps), 3)

        model.detect_anomalies(prediction)

        buckets = prediction.format_buckets()

        anomalies = buckets[0]['stats']['anomalies']
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies['avg_baz']['type'], 'high')

        anomalies = buckets[1]['stats']['anomalies']
        self.assertEqual(len(anomalies), 0)

        anomalies = buckets[2]['stats']['anomalies']
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies['avg_baz']['type'], 'low')




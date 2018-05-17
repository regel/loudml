import datetime
import logging
import math
import os
import random
import unittest

import numpy as np

logging.getLogger('tensorflow').disabled = True

from loudml.randevents import (
    FlatEventGenerator,
    SinEventGenerator,
)
from loudml.timeseries import (
    TimeSeriesModel,
    TimeSeriesPrediction,
)
from loudml.memdatasource import MemDataSource
from loudml.filestorage import TempStorage
from loudml.misc import (
    make_datetime,
    dt_get_weekday,
)

from loudml import (
    errors,
)

from loudml.api import Hook

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

class TestHook(Hook):
    def __init__(self):
        super().__init__(name='test')
        self.events = []

    def on_anomaly_start(self, model, dt, *args, **kwargs):
        self.events.append({
            'type': 'start',
            'dt': dt,
        })

    def on_anomaly_end(self, model, dt, *args, **kwargs):
        self.events.append({
            'type': 'end',
            'dt': dt,
        })


class TestTimes(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.source = MemDataSource()
        self.storage = TempStorage()

        self.model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=5,
            bucket_interval=20 * 60,
            interval=60,
            features=FEATURES,
            max_threshold=30,
            min_threshold=25,
            max_evals=5,
        ))

        self.generator = SinEventGenerator(avg=3, sigma=0.01)

        to_date = datetime.datetime.now().timestamp()

        # Be sure that date range is aligned
        self.to_date = math.floor(to_date / self.model.bucket_interval) * self.model.bucket_interval
        self.from_date = self.to_date - 3600 * 24 * 7 * 3

        for ts in self.generator.generate_ts(self.from_date, self.to_date, step=600):
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
        self.assertEqual(model.seasonality['weekday'], False)
        self.assertEqual(model.seasonality['daytime'], False)

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
        if self.model.is_trained:
            return

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
            max_threshold=30,
            min_threshold=25,
            max_evals=1,
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

        to_date = self.to_date
        from_date = to_date - 24 * 3600

        prediction = self.model.predict(self.source, from_date, to_date)

        expected = math.ceil(
            (to_date - from_date) / self.model.bucket_interval
        )

        # prediction.plot('count_foo')

        self.assertEqual(len(prediction.timestamps), expected)
        self.assertEqual(prediction.observed.shape, (expected, 2))
        self.assertEqual(prediction.predicted.shape, (expected, 2))

        for i in range(expected):
            self.assertAlmostEqual(
                prediction.observed[i][0],
                prediction.predicted[i][0],
                delta=2.5,
            )
            self.assertAlmostEqual(
                prediction.observed[i][1],
                prediction.predicted[i][1],
                delta=12,
            )

    @unittest.skip("FIXME")
    def test_predict_unaligned(self):
        self._require_training()

        # Unaligned date range
        to_date = self.to_date + self.model.bucket_interval / 4
        from_date = to_date

        prediction = self.model.predict(self.source, from_date, to_date)

        self.assertEqual(len(prediction.timestamps), 1)
        self.assertEqual(prediction.observed.shape, (1, 2))
        self.assertEqual(prediction.predicted.shape, (1, 2))

        self.assertAlmostEqual(
            prediction.observed[0][0],
            prediction.predicted[0][0],
            delta=2.0,
        )
        self.assertAlmostEqual(
            prediction.observed[0][1],
            prediction.predicted[0][1],
            delta=12,
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
                },
            ],
            max_threshold=30,
            min_threshold=25,
            max_evals=1,
        ))

        # train on all dataset
        model.train(source, hist_from, hist_to)
        self.assertTrue(model.is_trained)

        # predict on last 24h
        to_date = hist_to
        from_date = to_date - 3600 * 24
        prediction = model.predict(source, from_date, to_date)

        self.assertEqual(len(prediction.timestamps), 24)
        self.assertEqual(prediction.observed.shape, (24, 2))
        self.assertEqual(prediction.predicted.shape, (24, 2))

        # Adding this call to ensure detect_anomalies() can deal with nan
        self.model.detect_anomalies(prediction)

        self.assertEqual(prediction.format_series()['predicted']['avg_foo'][13:13+model.span],
                         [None] * model.span)

    def test_format_prediction(self):
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
            max_threshold=30,
            min_threshold=25,
            max_evals=1,
        ))

        timestamps = [self.to_date - i * model.bucket_interval for i in range(3)]

        prediction = TimeSeriesPrediction(
            model=model,
            timestamps=timestamps,
            observed=np.array([[1, 0.1], [np.nan, np.nan], [3, 0.3]]),
            predicted=np.array([[4, 0.4], [5, 0.5], [np.nan, np.nan]]),
        )

        self.assertEqual(
            prediction.format_series(),
            {
                'timestamps': timestamps,
                'observed': {
                    'count_foo': [1.0, 0.0, 3.0],
                    'avg_foo': [0.1, None, 0.3],
                },
                'predicted': {
                    'count_foo': [4.0, 5.0, 0.0],
                    'avg_foo': [0.4, 0.5, None],
                },
            }
        )

        self.assertEqual(
            prediction.format_buckets(),
            [
                {
                    'timestamp': timestamps[0],
                    'observed': {
                        'count_foo': 1.0,
                        'avg_foo': 0.1,
                    },
                    'predicted': {
                        'count_foo': 4.0,
                        'avg_foo': 0.4,
                    },
                },
                {
                    'timestamp': timestamps[1],
                    'observed': {
                        'count_foo': 0.0,
                        'avg_foo': None,
                    },
                    'predicted': {
                        'count_foo': 5.0,
                        'avg_foo': 0.5,
                    },
                },
                {
                    'timestamp': timestamps[2],
                    'observed': {
                        'count_foo': 3.0,
                        'avg_foo': 0.3,
                    },
                    'predicted': {
                        'count_foo': 0.0,
                        'avg_foo': None,
                    },
                },
            ]
        )

    def test_detect_anomalies(self):
        self._require_training()

        # Insert 2 buckets of normal data
        from_date = self.to_date
        to_date = from_date + 2 * self.model.bucket_interval

        for ts in self.generator.generate_ts(from_date, to_date, step=600):
            self.source.insert_times_data({
                'timestamp': ts,
                'foo': random.normalvariate(10, 1)
            })

        # Add abnormal data in the last bucket
        ano_from = from_date + self.model.bucket_interval
        ano_to = to_date
        generator = FlatEventGenerator(avg=4, sigma=0.01)

        for ts in generator.generate_ts(ano_from, ano_to, step=600):
            self.source.insert_times_data({
                'timestamp': ts,
                'foo': random.normalvariate(10, 1)
            })

        # Detect anomalies
        pred_to = ano_to
        pred_from = pred_to - 24 * 3 * self.model.bucket_interval
        prediction = self.model.predict(self.source, pred_from, pred_to)

        self.model.detect_anomalies(prediction)

        buckets = prediction.format_buckets()
        for bucket in buckets:
            stats = bucket.get('stats')
            self.assertIsNotNone(stats)
            self.assertIsNotNone(stats.get('mse'))
            self.assertIsNotNone(stats.get('score'))
            self.assertIsNotNone(stats.get('anomaly'))

        #print(prediction)
        #prediction.plot('count_foo')

        # First bucket is normal
        self.assertFalse(buckets[-2]['stats']['anomaly'])

        # Anomaly detected in second bucket
        self.assertTrue(buckets[-1]['stats']['anomaly'])

        anomalies = prediction.get_anomalies()
        self.assertEqual(anomalies, [buckets[-1]])

    def test_span_auto(self):
        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span="auto",
            bucket_interval=3600,
            interval=60,
            features=[
                {
                   'name': 'count_foo',
                   'metric': 'count',
                   'field': 'foo',
                   'default': 0,
                },
            ],
            max_threshold=30,
            min_threshold=25,
            max_evals=50,
        ))

        self.assertEqual(model.span, "auto")
        model.train(self.source, self.from_date, self.to_date)
        self.assertTrue(4 < model.span < 20)

    def test_daytime_model(self):
        source = MemDataSource()
        to_date = datetime.datetime.now().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ).timestamp()

        # Generate N days of data
        nb_days = 90
        hist_to = to_date
        hist_from = to_date - 3600 * 24 * nb_days
        ts = hist_from
        for i in range(nb_days):
            # [0h-12h[ data. Regular point at 2 AM
            for j in range(12):
                if j == 2:
                    source.insert_times_data({
                        'timestamp': ts,
                        'foo': j,
                    })
                ts += 3600

            # No data for [12h, 24h[
            for j in range(12):
                ts += 3600

        # insert anomaly (no data point expected in this time range)
        ts = hist_to - 3600 * 6
        source.insert_times_data({
            'timestamp': ts,
            'foo': j,
        })

        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=3,
            bucket_interval=3600,
            interval=60,
            seasonality={
                'daytime': True,
            },
            features=[
                {
                   'name': 'count_foo',
                   'metric': 'count',
                   'field': 'foo',
                   'default': 0,
                },
            ],
            max_threshold=30,
            min_threshold=25,
            max_evals=1,
        ))
        self.assertTrue(model.seasonality.get('daytime'), True)

        # train on N-1 days
        model.train(source, hist_from, hist_to - 3600 * 24)
        self.assertTrue(model.is_trained)

        # predict on last 24h
        to_date = hist_to
        from_date = to_date - 3600 * 24
        prediction = model.predict(source, from_date, to_date)

        self.assertEqual(len(prediction.timestamps), 24)
        self.assertEqual(prediction.observed.shape, (24, 1))
        self.assertEqual(prediction.predicted.shape, (24, 1))

        model.detect_anomalies(prediction)
        buckets = prediction.format_buckets()
        # Anomaly MUST NOT be detected in bucket[-22] (the 2 AM point)
        self.assertFalse(buckets[-22]['stats']['anomaly'])
        # Anomaly MUST be detected in bucket[-6]
        self.assertTrue(buckets[-6]['stats']['anomaly'])
        self.assertAlmostEqual(100, buckets[-6]['stats']['score'], delta=10)

    def test_not_daytime_model(self):
        source = MemDataSource()
        to_date = datetime.datetime.now().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ).timestamp()

        # Generate N days of data
        nb_days = 90
        hist_to = to_date
        hist_from = to_date - 3600 * 24 * nb_days
        ts = hist_from
        for i in range(nb_days):
            # [0h-12h[ data. Regular point at 2 AM
            for j in range(12):
                if j == 2:
                    source.insert_times_data({
                        'timestamp': ts,
                        'foo': j,
                    })
                ts += 3600

            # No data for [12h, 24h[
            for j in range(12):
                ts += 3600

        # insert anomaly (no data point expected in this time range)
        ts = hist_to - 3600 * 6
        source.insert_times_data({
            'timestamp': ts,
            'foo': j,
        })

        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=3,
            bucket_interval=3600,
            interval=60,
            seasonality={
                'daytime': False,
            },
            features=[
                {
                   'name': 'count_foo',
                   'metric': 'count',
                   'field': 'foo',
                   'default': 0,
                },
            ],
            max_threshold=30,
            min_threshold=25,
            max_evals=1,
        ))

        # train on N-1 days
        model.train(source, hist_from, hist_to - 3600 * 24)
        self.assertTrue(model.is_trained)

        # predict on last 24h
        to_date = hist_to
        from_date = to_date - 3600 * 24
        prediction = model.predict(source, from_date, to_date)

        self.assertEqual(len(prediction.timestamps), 24)
        self.assertEqual(prediction.observed.shape, (24, 1))
        self.assertEqual(prediction.predicted.shape, (24, 1))

        model.detect_anomalies(prediction)
        buckets = prediction.format_buckets()
        # Anomaly MUST be detected in bucket[-22] (the 2 AM point)
        self.assertTrue(buckets[-22]['stats']['anomaly'])
        self.assertAlmostEqual(100, buckets[-22]['stats']['score'], delta=10)
        # Anomaly MUST be detected in bucket[-6]
        self.assertTrue(buckets[-6]['stats']['anomaly'])
        self.assertAlmostEqual(100, buckets[-6]['stats']['score'], delta=10)

    def test_weekday_model(self):
        source = MemDataSource()
        to_date = datetime.datetime.now().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=datetime.timezone.utc,
        ).timestamp()

        nb_days = 200
        hist_to = to_date
        hist_from = to_date - 3600 * 24 * nb_days
        ts = hist_from
        for i in range(nb_days):
            dt = make_datetime(ts)

            if dt_get_weekday(dt) <= 5:
                # Workday
                value = 1
            else:
                # Week-end
                value = 0

            for i in range(24):
                source.insert_times_data({
                    'timestamp': ts,
                    'foo': value,
                })
                ts += 3600

        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=12,
            bucket_interval=4 * 3600,
            interval=60,
            seasonality={
                'weekday': True,
            },
            features=[
                {
                   'name': 'avg_foo',
                   'metric': 'avg',
                   'field': 'foo',
                   'default': 0,
                },
            ],
            max_threshold=30,
            min_threshold=25,
            max_evals=5,
        ))

        model.train(source, hist_from, hist_to)
        self.assertTrue(model.is_trained)

        for i in range(7):
            to_date = hist_to - i * 3600 * 24
            from_date = to_date - 3600 * 24 * 7
            prediction = model.predict(source, from_date, to_date)
            model.detect_anomalies(prediction)
            self.assertEqual(len(prediction.get_anomalies()), 0)

    def test_weekday_daytime_model(self):
        source = MemDataSource()
        to_date = datetime.datetime.now().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=datetime.timezone.utc,
        ).timestamp()

        # Generate N days of data
        nb_days = 500
        hist_to = to_date
        hist_from = to_date - 3600 * 24 * nb_days
        ts = hist_from
        for i in range(nb_days):
            dt = make_datetime(ts)

            if dt_get_weekday(dt) <= 5:
                # Workday
                value = 1
            else:
                # Week-end
                value = 0

            # [8h-18h[
            ts += 3600 * 8
            for j in range(8, 18):
                source.insert_times_data({
                    'timestamp': ts,
                    'foo': value,
                })
                ts += 3600

            ts += 3600 * 6

        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=6,
            bucket_interval=3600,
            interval=60,
            seasonality={
                'daytime': True,
                'weekday': True,
            },
            features=[
                {
                   'name': 'count_foo',
                   'metric': 'avg',
                   'field': 'foo',
                   'default': 0,
                },
            ],
            max_threshold=30,
            min_threshold=25,
            max_evals=1,
        ))

        model.train(source, hist_from, hist_to)
        self.assertTrue(model.is_trained)

        for i in range(7):
            to_date = hist_to - i * 3600 * 24
            from_date = to_date - 3600 * 24 * 7
            prediction = model.predict(source, from_date, to_date)
            self.assertEqual(len(prediction.timestamps), 24 * 7)
            model.detect_anomalies(prediction)
            self.assertEqual(len(prediction.get_anomalies()), 0)

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

        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=6,
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
            max_threshold=50,
            min_threshold=45,
            max_evals=1,
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

        hook = TestHook()

        model.detect_anomalies(prediction, hooks=[hook])

        self.assertEqual(len(hook.events), 2)
        event0, event1 = hook.events
        self.assertEqual(event0['type'], 'start')
        self.assertEqual(event1['type'], 'end')
        self.assertGreaterEqual(
            (event1['dt'] - event0['dt']).seconds,
            6 * 3600,
        )

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
                    'foo': random.randrange(45, 55),
                    'bar': random.randrange(45, 55),
                    'baz': random.randrange(45, 55),

                })
                ts += 3600

        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=6,
            bucket_interval=3600,
            interval=60,
            features=[
                {
                   'name': 'avg_foo',
                   'metric': 'avg',
                   'field': 'foo',
                   'anomaly_type': 'low',
                },
                {
                   'name': 'avg_bar',
                   'metric': 'avg',
                   'field': 'bar',
                   'anomaly_type': 'high',
                },
                {
                   'name': 'avg_baz',
                   'metric': 'avg',
                   'field': 'baz',
                   'anomaly_type': 'low_high',
                },
            ],
            max_threshold=30,
            min_threshold=25,
            max_evals=1,
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
                'bar': values[1],
                'baz': values[2],
            })
            ts += 3600

        prediction = model.predict(source, hist_to, ts)
        self.assertEqual(len(prediction.timestamps), 3)

        model.detect_anomalies(prediction)

        buckets = prediction.format_buckets()

        anomalies = buckets[0]['stats']['anomalies']
        self.assertEqual(len(anomalies), 2)
        self.assertEqual(anomalies['avg_foo']['type'], 'low')
        self.assertEqual(anomalies['avg_baz']['type'], 'high')

        anomalies = buckets[1]['stats']['anomalies']
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies['avg_bar']['type'], 'high')

        anomalies = buckets[2]['stats']['anomalies']
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies['avg_baz']['type'], 'low')

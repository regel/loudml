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
)
from loudml.timeseries import (
    TimeSeriesModel,
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

        self.generator = SinEventGenerator(base=3, amplitude=3, sigma=0.01)

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
            features=FEATURES[:1],
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


    def test_forecast(self):
        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=21,
            forecast=7,
            bucket_interval=20 * 60,
            interval=60,
            features=FEATURES[:1],
            threshold=30,
            max_evals=10,
        ))
        source = MemDataSource()
        generator = SinEventGenerator(base=3, amplitude=3, sigma=0.01)

        # Align date range to day interval
        to_date = datetime.datetime.now().timestamp()
        to_date = math.floor(to_date / (3600*24)) * (3600*24)
        from_date = to_date - 3600 * 24 * 7 * 3
        for ts in generator.generate_ts(from_date, to_date, step=600):
            source.insert_times_data({
                'timestamp': ts,
                'foo': random.normalvariate(10, 1)
            })

        model.train(source, from_date, to_date)
        prediction = model.predict(source, from_date, to_date)

        from_date = to_date - model.bucket_interval 
        to_date = from_date + 48 * 3600
        forecast = model.forecast(source, from_date, to_date)

        expected = math.ceil(
            (to_date - from_date) / model.bucket_interval
        )

        forecast.plot('count_foo')
        import matplotlib.pylab as plt
        plt.rcParams["figure.figsize"] = (17, 9)
        y_values = prediction.observed[:,0]
        plt.plot(range(1,1+len(y_values)), y_values, "--")
        z_values = forecast.predicted[:,0]
        plt.plot(range(len(y_values), len(y_values)+len(z_values)), z_values, ":")
        plt.show()

        self.assertEqual(len(forecast.timestamps), expected)
        self.assertEqual(forecast.observed.shape, (expected, 1))
        self.assertEqual(forecast.predicted.shape, (expected, 1))

        all_nans = np.empty((expected, 1), dtype=float)
        all_nans[:] = np.nan
        self.assertEqual(nan_equal(forecast.observed, all_nans), True)

        forecast_head = np.array([[9.15], [9.64], [10.06], [10.44], [10.71]])
        forecast_tail = np.array([[8.20], [8.83], [9.30], [9.71], [10.10]])

#        print(forecast.predicted)
        delta = 2.8
        forecast_good = np.abs(forecast.predicted[:len(forecast_head)] - forecast_head) <= delta
        # print(forecast_head)
        # print(forecast.predicted[:len(forecast_head)])
        # print(forecast_good)
        self.assertEqual(np.all(forecast_good), True)
        forecast_good = np.abs(forecast.predicted[-len(forecast_tail):] - forecast_tail) <= delta
        # print(forecast_tail)
        # print(forecast.predicted[-len(forecast_tail):])
        # print(forecast_good)
        self.assertEqual(np.all(forecast_good), True)

    def test_forecast2(self):
        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=21,
            forecast=7,
            bucket_interval=20 * 60,
            interval=60,
            features=FEATURES,
            threshold=30,
            max_evals=10,
        ))
        source = MemDataSource()
        generator = SinEventGenerator(base=3, amplitude=3, sigma=0.01)
        # Align date range to day interval
        to_date = datetime.datetime.now().timestamp()
        to_date = math.floor(to_date / (3600*24)) * (3600*24)
        from_date = to_date - 3600 * 24 * 7 * 3
        for ts in generator.generate_ts(from_date, to_date, step=600):
            source.insert_times_data({
                'timestamp': ts,
                'foo': random.normalvariate(10, 1)
            })

        model.train(source, from_date, to_date)

        # Verify that predict() is still functional when forecast>1
        prediction = model.predict(source, to_date - 48 * 3600, to_date)

        expected = math.ceil(
            (48 * 3600) / model.bucket_interval
        )
        self.assertEqual(len(prediction.timestamps), expected)
        self.assertEqual(prediction.observed.shape, (expected, 2))
        self.assertEqual(prediction.predicted.shape, (expected, 2))

        for i in range(expected):
            self.assertAlmostEqual(
                prediction.observed[i][0],
                prediction.predicted[i][0],
                delta=2.8,
            )
            self.assertAlmostEqual(
                prediction.observed[i][1],
                prediction.predicted[i][1],
                delta=5,
            )

        from_date = to_date - model.bucket_interval 
        to_date = from_date + 48 * 3600
        forecast = model.forecast(source, from_date, to_date)
        expected = math.ceil(
            (to_date - from_date) / model.bucket_interval
        )

#        forecast.plot('count_foo')
#        import matplotlib.pylab as plt
#        plt.rcParams["figure.figsize"] = (17, 9)
#        y_values = prediction.observed[:,0]
#        plt.plot(range(1,1+len(y_values)), y_values, "--")
#        z_values = forecast.predicted[:,0]
#        plt.plot(range(len(y_values), len(y_values)+len(z_values)), z_values, ":")
#        plt.show()

        self.assertEqual(len(forecast.timestamps), expected)
        self.assertEqual(forecast.observed.shape, (expected, 2))
        self.assertEqual(forecast.predicted.shape, (expected, 2))

        all_nans = np.empty((expected, 2), dtype=float)
        all_nans[:] = np.nan
        self.assertEqual(nan_equal(forecast.observed, all_nans), True)

        forecast_head = np.array([9.15, 9.64, 10.06, 10.44, 10.71])
        forecast_tail = np.array([8.20, 8.83, 9.30, 9.71, 10.10])

#        print(forecast.predicted[:,0])
        # Verify forecast(count_foo) feature, must have sin shape
        delta = 2.8
        forecast_good = np.abs(forecast.predicted[:len(forecast_head),0] - forecast_head) <= delta
#        print(forecast.predicted[:len(forecast_head),0])
#        print(forecast_head)
#        print(forecast_good)
        self.assertEqual(np.all(forecast_good), True)
        forecast_good = np.abs(forecast.predicted[-len(forecast_tail):,0] - forecast_tail) <= delta
#        print(forecast.predicted[-len(forecast_tail):,0])
#        print(forecast_tail)
#        print(forecast_good)
        self.assertEqual(np.all(forecast_good), True)

        # Verify forecast(avg_foo) feature, must be const noise centered around value=10
        forecast_head = np.array([ 10, 10, 10, 10, 10])
        forecast_tail = np.array([ 10, 10, 10, 10, 10])
        delta = 0.5
        forecast_good = np.abs(forecast.predicted[:len(forecast_head),1] - forecast_head) <= delta
#       print(forecast.predicted[:len(forecast_head),1])
#       print(forecast_head)
#        print(forecast_good)
        self.assertEqual(np.all(forecast_good), True)
        forecast_good = np.abs(forecast.predicted[-len(forecast_tail):,1] - forecast_tail) <= delta
#        print(forecast.predicted[-len(forecast_tail):,1])
#        print(forecast_tail)
#        print(forecast_good)
        self.assertEqual(np.all(forecast_good), True)

    def test_forecast_daytime(self):
        source = MemDataSource()
        generator = SinEventGenerator(base=3, amplitude=3, sigma=0.01)
        # Align date range to day interval
        to_date = datetime.datetime.now().timestamp()
        to_date = math.floor(to_date / (3600*24)) * (3600*24)
        from_date = to_date - 3600 * 24 * 7 * 3

        for i, ts in enumerate(generator.generate_ts(from_date, to_date, step=600)):
            dt = make_datetime(ts)
            val = random.normalvariate(10, 1)
            if dt_get_daytime(dt) < 6 or dt_get_daytime(dt) > 22:
                val = val / 2
                if (i % 2) == 0:
                    source.insert_times_data({
                        'timestamp': ts,
                        'foo': val,
                    })
            else:
                source.insert_times_data({
                    'timestamp': ts,
                    'foo': val,
                })

        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=7,
            forecast=5,
            seasonality={
                'daytime': True,
            },
            bucket_interval=20 * 60,
            interval=60,
            features=FEATURES[:1],
            threshold=30,
            max_evals=10,
        ))

        model.train(source, from_date, to_date)

        prediction = model.predict(source, from_date, to_date)
        # prediction.plot('count_foo')

        expected = math.ceil(
            (to_date - from_date) / model.bucket_interval
        )

        self.assertEqual(len(prediction.timestamps), expected)
        self.assertEqual(prediction.observed.shape, (expected, 1))
        self.assertEqual(prediction.predicted.shape, (expected, 1))

        from_date = to_date - model.bucket_interval 
        to_date = from_date + 48 * 3600
        forecast = model.forecast(source, from_date, to_date)

        expected = math.ceil(
            (to_date - from_date) / model.bucket_interval
        )

        # forecast.plot('count_foo')

        self.assertEqual(len(forecast.timestamps), expected)
        self.assertEqual(forecast.observed.shape, (expected, 1))
        self.assertEqual(forecast.predicted.shape, (expected, 1))

        all_nans = np.empty((expected, 1), dtype=float)
        all_nans[:] = np.nan
        self.assertEqual(nan_equal(forecast.observed, all_nans), True)

        #import matplotlib.pylab as plt
        #plt.rcParams["figure.figsize"] = (17, 9)
        #y_values = prediction.observed[:,0]
        #plt.plot(range(1,1+len(y_values)), y_values, "--")
        #z_values = forecast.predicted[:,0]
        #plt.plot(range(len(y_values), len(y_values)+len(z_values)), z_values, ":")
        #plt.show()

        forecast_head = np.array([3.95, 4.47, 4.70, 4.85, 5.31])
        forecast_tail = np.array([4.17, 3.68, 3.97, 3.93, 4.07])

        # Verify forecast(count_foo) feature, must have sin shape
        delta = 1.0
        forecast_good = np.abs(forecast.predicted[:len(forecast_head),0] - forecast_head) <= delta
        # print(forecast.predicted[:len(forecast_head),0])
        # print(forecast_head)
        # print(forecast_good)
        self.assertEqual(np.all(forecast_good), True)
        forecast_good = np.abs(forecast.predicted[-len(forecast_tail):,0] - forecast_tail) <= delta
        # print(forecast.predicted[-len(forecast_tail):,0])
        # print(forecast_tail)
        # print(forecast_good)
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
        model.detect_anomalies(prediction)

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
        generator = FlatEventGenerator(base=4, sigma=0.01)

        for ts in generator.generate_ts(ano_from, ano_to, step=600):
            self.source.insert_times_data({
                'timestamp': ts,
                'foo': random.normalvariate(10, 1)
            })

        # Detect anomalies
        pred_to = to_date
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
            forecast=5,
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
            threshold=30,
            max_evals=10,
        ))

        self.assertEqual(model.span, "auto")
        model.train(self.source, self.from_date, self.to_date)
        self.assertTrue(17 <= model._span <= 19)

    def test_daytime_model(self):
        source = MemDataSource()

        # Generate N days of data
        nb_days = 90
        to_date = datetime.datetime.now().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ).timestamp()
        from_date = to_date - 3600 * 24 * nb_days
        generator = SinEventGenerator(base=3, amplitude=3, sigma=0.01)

        # Regular data
        for ts in self.generator.generate_ts(self.from_date, self.to_date, step=60):
            source.insert_times_data({
                'timestamp': ts,
                'foo': random.normalvariate(10, 1)
            })

        # Insert anomaly at 02:00-04:00 on the last day
        ano_from = to_date - 22 * 3600
        ano_to = ano_from + 2 * 3600

        generator = FlatEventGenerator(base=4, sigma=0.01)

        for ts in generator.generate_ts(ano_from, ano_to, step=60):
            source.insert_times_data({
                'timestamp': ts,
                'foo': random.normalvariate(10, 1)
            })

        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=8,
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
            max_evals=10,
        ))
        self.assertTrue(model.seasonality.get('daytime'), True)

        # train on N-1 days
        model.train(source, from_date, to_date - 3600 * 24)
        self.assertTrue(model.is_trained)

        # predict on last 24h
        pred_to = to_date
        pred_from = to_date - 3600 * 24 * 4
        prediction = model.predict(source, pred_from, pred_to)

        prediction.plot('count_foo')

        self.assertEqual(len(prediction.timestamps), 24)
        self.assertEqual(prediction.observed.shape, (24, 1))
        self.assertEqual(prediction.predicted.shape, (24, 1))

        model.detect_anomalies(prediction)
        buckets = prediction.format_buckets()


        import json
        print(json.dumps(buckets, indent=4))

        # No anomaly MUST be detected on 00:00-02:00
        self.assertFalse(buckets[0]['stats']['anomaly'])
        self.assertFalse(buckets[1]['stats']['anomaly'])

        # Anomaly MUST be detected on 02:00-04:00
        self.assertTrue(buckets[2]['stats']['anomaly'])
        self.assertAlmostEqual(100, buckets[2]['stats']['score'], delta=10)
        self.assertTrue(buckets[3]['stats']['anomaly'])
        self.assertAlmostEqual(100, buckets[3]['stats']['score'], delta=10)

        # No anomaly after on 04:00
        self.assertFalse(buckets[4]['stats']['anomaly'])
        self.assertFalse(buckets[4]['stats']['anomaly'])

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


    def test_model_dict(self):
        """
        This test is meant to detect internal errors,
        not to test the actual predicted values.
        """
        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=5,
            forecast=1,
            bucket_interval=20 * 60,
            interval=60,
            seasonality={
                'daytime': False,
                'weekday': False,
            },
            features=dict(
                i=[
                  {
                   'name': 'avg_foo',
                   'metric': 'avg',
                   'field': 'foo',
                   'default': 0,
                  },
                ],
                o=[
                  {
                   'name': 'count_foo',
                   'metric': 'count',
                   'field': 'foo',
                   'default': 0,
                  },
                ],
            ),
            threshold=30,
            max_evals=1,
        ))
        model.train(self.source, self.from_date, self.to_date)
        self.assertTrue(model.is_trained)

        from_date = self.to_date - 48 * 3600
        to_date = self.to_date
        prediction = model.predict(self.source, from_date, to_date)

        expected = math.ceil(
            (to_date - from_date) / model.bucket_interval
        )

        self.assertEqual(len(prediction.timestamps), expected)
        self.assertEqual(prediction.observed.shape, (expected, 2))
        self.assertEqual(prediction.predicted.shape, (expected, 2))

        series = prediction.format_series()
        self.assertTrue(isinstance(series['timestamps'], list))
        self.assertEqual(len(series['observed'].keys()), 1)
        self.assertTrue(isinstance(series['observed']['avg_foo'], list))
        self.assertEqual(len(series['predicted'].keys()), 1)
        self.assertTrue(isinstance(series['predicted']['count_foo'], list))

        from_date = self.to_date - model.bucket_interval
        to_date = self.to_date + 24 * 3600
        prediction = model.forecast(self.source, from_date, to_date)

        expected = math.ceil(
            (to_date - from_date) / model.bucket_interval
        )

        self.assertEqual(len(prediction.timestamps), expected)
        self.assertEqual(prediction.observed.shape, (expected, 2))
        self.assertEqual(prediction.predicted.shape, (expected, 2))

        for bucket in prediction.format_buckets():
            self.assertTrue(isinstance(bucket['timestamp'], float))
            self.assertTrue(isinstance(bucket['observed']['avg_foo'], float))
            self.assertTrue(isinstance(bucket['predicted']['count_foo'], float))

    def test_model_dict2(self):
        """
        This test is meant to detect internal errors,
        not to test the actual predicted values.
        """
        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=5,
            forecast=1,
            bucket_interval=20 * 60,
            interval=60,
            seasonality={
                'daytime': False,
                'weekday': False,
            },
            features=dict(
                io=[
                  {
                   'name': 'avg_foo',
                   'metric': 'avg',
                   'field': 'foo',
                   'default': 0,
                  },
                ],
                o=[
                  {
                   'name': 'count_foo',
                   'metric': 'count',
                   'field': 'foo',
                   'default': 0,
                  },
                ],
            ),
            threshold=30,
            max_evals=1,
        ))
        model.train(self.source, self.from_date, self.to_date)
        self.assertTrue(model.is_trained)

        from_date = self.to_date - 48 * 3600
        to_date = self.to_date
        prediction = model.predict(self.source, from_date, to_date)

        expected = math.ceil(
            (to_date - from_date) / model.bucket_interval
        )

        self.assertEqual(len(prediction.timestamps), expected)
        self.assertEqual(prediction.observed.shape, (expected, 2))
        self.assertEqual(prediction.predicted.shape, (expected, 2))

        series = prediction.format_series()
        self.assertTrue(isinstance(series['timestamps'], list))
        self.assertEqual(len(series['observed'].keys()), 1)
        self.assertTrue(isinstance(series['observed']['avg_foo'], list))
        self.assertEqual(len(series['predicted'].keys()), 2)
        self.assertTrue(isinstance(series['predicted']['count_foo'], list))
        self.assertTrue(isinstance(series['predicted']['avg_foo'], list))

        from_date = self.to_date - model.bucket_interval
        to_date = self.to_date + 24 * 3600
        prediction = model.forecast(self.source, from_date, to_date)

        expected = math.ceil(
            (to_date - from_date) / model.bucket_interval
        )

        self.assertEqual(len(prediction.timestamps), expected)
        self.assertEqual(prediction.observed.shape, (expected, 2))
        self.assertEqual(prediction.predicted.shape, (expected, 2))

        for bucket in prediction.format_buckets():
            self.assertTrue(isinstance(bucket['timestamp'], float))
            self.assertTrue(isinstance(bucket['observed']['avg_foo'], float))
            self.assertTrue(isinstance(bucket['predicted']['count_foo'], float))
            self.assertTrue(isinstance(bucket['predicted']['avg_foo'], float))

    def test_model_dict3(self):
        """
        This test is meant to detect internal errors,
        not to test the actual predicted values.
        """
        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=5,
            forecast=1,
            bucket_interval=20 * 60,
            interval=60,
            seasonality={
                'daytime': True,
                'weekday': False,
            },
            features=dict(
                i=[
                  {
                   'name': 'avg_foo',
                   'metric': 'avg',
                   'field': 'foo',
                   'default': 0,
                  },
                ],
                o=[
                  {
                   'name': 'count_foo',
                   'metric': 'count',
                   'field': 'foo',
                   'default': 0,
                  },
                ],
            ),
            threshold=30,
            max_evals=1,
        ))
        model.train(self.source, self.from_date, self.to_date)
        self.assertTrue(model.is_trained)

        from_date = self.to_date - 48 * 3600
        to_date = self.to_date
        prediction = model.predict(self.source, from_date, to_date)

        expected = math.ceil(
            (to_date - from_date) / model.bucket_interval
        )

        self.assertEqual(len(prediction.timestamps), expected)
        self.assertEqual(prediction.observed.shape, (expected, 2))
        self.assertEqual(prediction.predicted.shape, (expected, 2))

        series = prediction.format_series()
        self.assertTrue(isinstance(series['timestamps'], list))
        self.assertEqual(len(series['observed'].keys()), 1)
        self.assertTrue(isinstance(series['observed']['avg_foo'], list))
        self.assertEqual(len(series['predicted'].keys()), 1)
        self.assertTrue(isinstance(series['predicted']['count_foo'], list))

        from_date = self.to_date - model.bucket_interval
        to_date = self.to_date + 24 * 3600
        prediction = model.forecast(self.source, from_date, to_date)

        expected = math.ceil(
            (to_date - from_date) / model.bucket_interval
        )

        self.assertEqual(len(prediction.timestamps), expected)
        self.assertEqual(prediction.observed.shape, (expected, 2))
        self.assertEqual(prediction.predicted.shape, (expected, 2))

        for bucket in prediction.format_buckets():
            self.assertTrue(isinstance(bucket['timestamp'], float))
            self.assertTrue(isinstance(bucket['observed']['avg_foo'], float))
            self.assertTrue(isinstance(bucket['predicted']['count_foo'], float))

    def test_model_dict4(self):
        """
        This test is meant to detect internal errors,
        not to test the actual predicted values.
        """
        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=5,
            forecast=1,
            bucket_interval=20 * 60,
            interval=60,
            seasonality={
                'daytime': True,
                'weekday': True,
            },
            features=dict(
                io=[
                  {
                   'name': 'avg_foo',
                   'metric': 'avg',
                   'field': 'foo',
                   'default': 0,
                  },
                ],
                o=[
                  {
                   'name': 'count_foo',
                   'metric': 'count',
                   'field': 'foo',
                   'default': 0,
                  },
                ],
            ),
            threshold=30,
            max_evals=1,
        ))
        model.train(self.source, self.from_date, self.to_date)
        self.assertTrue(model.is_trained)

        from_date = self.to_date - 48 * 3600
        to_date = self.to_date
        prediction = model.predict(self.source, from_date, to_date)

        expected = math.ceil(
            (to_date - from_date) / model.bucket_interval
        )

        self.assertEqual(len(prediction.timestamps), expected)
        self.assertEqual(prediction.observed.shape, (expected, 2))
        self.assertEqual(prediction.predicted.shape, (expected, 2))

        series = prediction.format_series()
        self.assertTrue(isinstance(series['timestamps'], list))
        self.assertEqual(len(series['observed'].keys()), 1)
        self.assertTrue(isinstance(series['observed']['avg_foo'], list))
        self.assertEqual(len(series['predicted'].keys()), 2)
        self.assertTrue(isinstance(series['predicted']['count_foo'], list))
        self.assertTrue(isinstance(series['predicted']['avg_foo'], list))

        from_date = self.to_date - model.bucket_interval
        to_date = self.to_date + 24 * 3600
        prediction = model.forecast(self.source, from_date, to_date)

        expected = math.ceil(
            (to_date - from_date) / model.bucket_interval
        )

        self.assertEqual(len(prediction.timestamps), expected)
        self.assertEqual(prediction.observed.shape, (expected, 2))
        self.assertEqual(prediction.predicted.shape, (expected, 2))

        for bucket in prediction.format_buckets():
            self.assertTrue(isinstance(bucket['timestamp'], float))
            self.assertTrue(isinstance(bucket['observed']['avg_foo'], float))
            self.assertTrue(isinstance(bucket['predicted']['count_foo'], float))
            self.assertTrue(isinstance(bucket['predicted']['avg_foo'], float))

    def test_model_nl(self):
        """
        Test real world data
        """
        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=3,
            forecast=1,
            bucket_interval="10m",
            interval=60,
            seasonality={
                'daytime': False,
                'weekday': False,
            },
            features=dict(
                i=[
                  {
                   'name': 'mean_CO2',
                   'metric': 'avg',
                   'field': 'CO2',
                  },
                  {
                   'name': 'mean_Noise',
                   'metric': 'avg',
                   'field': 'Noise',
                  },
                ],
                o=[
                  {
                   'name': 'mean_NrP',
                   'metric': 'avg',
                   'field': 'NrP',
                   'default': 0,
                  },
                ],
            ),
            threshold=30,
            max_evals=10,
        ))

        source = MemDataSource()

        csv_path = os.path.join(
            os.path.dirname(__file__),
            'resources',
            'nl.csv.gz',
        )
        source.load_csv(
            csv_path,
            encoding="utf-8",
            timestamp_field="DT",
            delimiter=";",
        )
        from_date = "2018-03-12 00:00"
        to_date = "2018-03-19 00:00"
        model.train(source, from_date, to_date)
        self.assertTrue(model.is_trained)

        prediction = model.predict(source, from_date, to_date)

        expected = math.ceil(
            (make_ts(to_date) - make_ts(from_date)) / model.bucket_interval
        )

        self.assertEqual(len(prediction.timestamps), expected)
        self.assertEqual(prediction.observed.shape, (expected, 3))
        self.assertEqual(prediction.predicted.shape, (expected, 3))

        series = prediction.format_series()
        self.assertTrue(isinstance(series['timestamps'], list))
        self.assertEqual(len(series['observed'].keys()), 2)
        self.assertTrue(isinstance(series['observed']['mean_CO2'], list))
        self.assertTrue(isinstance(series['observed']['mean_Noise'], list))
        self.assertEqual(len(series['predicted'].keys()), 1)
        self.assertTrue(isinstance(series['predicted']['mean_NrP'], list))

        features=[Feature(name='mean_NrP', metric='avg', field='NrP', default=0)]
        rows = source._get_times_data(bucket_interval=model.bucket_interval,
                                      from_date=from_date,
                                      to_date=to_date,
                                      features=features)
        y_values = []
        for _, y, _ in rows:
            y_values.append(y)

        y_values = np.array(y_values)
        z_values = prediction.predicted[:,0]
        for i, z in enumerate(z_values):
            z_values[i] = int(z)

        self.assertTrue(np.mean(y_values - z_values) <= 0.4)

#        import matplotlib.pylab as plt
#        plt.rcParams["figure.figsize"] = (17, 9)
#        plt.plot(y_values, "--")
#        plt.plot(z_values, ":")
#        plt.show()

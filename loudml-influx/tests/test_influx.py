import loudml.vendor

import datetime
from datetime import timezone
import logging
import numpy as np
import os
import random
import time
import unittest
import math
import json

logging.getLogger('tensorflow').disabled = True

import loudml.errors as errors
try:
    import loudml.test
except ImportError as exn:
    # ignore fingerprint import error
    print("warning:", exn)

from loudml.misc import (
    escape_quotes,
    escape_doublequotes,
    nan_to_none,
)

from loudml.influx import (
    _build_time_predicates,
    _build_tags_predicates,
    InfluxDataSource,
)

from loudml.timeseries import TimeSeriesModel
from loudml.randevents import SinEventGenerator
from loudml.filestorage import TempStorage


FEATURES = [
    {
        'name': 'avg_foo',
        'metric': 'avg',
        'measurement': 'measure1',
        'field': 'foo',
        'default': 0,
    },
    {
        'name': 'count_bar',
        'metric': 'count',
        'measurement': 'measure2',
        'field': 'bar',
        'default': 0,
    },
    {
        'name': 'avg_baz',
        'metric': 'avg',
        'measurement': 'measure1',
        'field': 'baz',
        'match_all': [
            {'tag': 'mytag', 'value': 'myvalue'},
        ],
        'default': 0,
    },
]

FEATURES_MATCH_ALL_TAG1 = [
    {
        'name': 'avg_baz',
        'metric': 'avg',
        'measurement': 'measure3',
        'field': 'baz',
        'match_all': [
            {'tag': 'tag_kw', 'value': 'tag1'},
        ],
    },
]
FEATURES_MATCH_ALL_TAG2 = [
    {
        'name': 'avg_baz',
        'metric': 'avg',
        'measurement': 'measure3',
        'field': 'baz',
        'match_all': [
            {'tag': 'tag_kw', 'value': 'tag2'},
            {'tag': 'tag_int', 'value': 7},
            {'tag': 'tag_bool', 'value': True},
        ],
    },
]

if 'INFLUXDB_ADDR' in os.environ:
    ADDR = os.environ['INFLUXDB_ADDR']
else:
    ADDR = 'localhost'

class TestInfluxQuick(unittest.TestCase):
    def setUp(self):
        bucket_interval = 3

        t0 = int(datetime.datetime.now().timestamp())

        # XXX Bucket returned by InfluxDB are aligne on modulo(bucket_interval), that's why
        # timestamp must be aligned for unit tests.
        t0 -= t0 % bucket_interval

        self.t0 = t0

        self.db = 'test-{}'.format(t0)
        logging.info("creating database %s", self.db)
        self.source = InfluxDataSource({
            'name': 'test',
            'addr': ADDR,
            'database': self.db,
        })
        self.source.drop()
        self.source.init()

        self.model = TimeSeriesModel(dict(
            name="test-model",
            offset=30,
            span=300,
            bucket_interval=3,
            interval=60,
            features=FEATURES,
            threshold=30,
        ))

        data = [
            # (foo, bar, timestamp)
            (1, 33, t0 - 1), # excluded
            (2, 120, t0), (3, 312, t0 + 1),
            # empty
            (4, 18, t0 + 7),
            (5, 78, t0 + 9), # excluded
        ]
        for foo, bar, ts in data:
            self.source.insert_times_data(
                measurement='measure1',
                ts=ts,
                data={
                    'foo': foo,
                }
            )
            self.source.insert_times_data(
                measurement='measure2',
                ts=ts,
                data={
                    'bar': bar,
                }
            )
            self.source.insert_times_data(
                measurement='measure3',
                ts=ts,
                tags={
                    'tag_kw': 'tag1',
                    'tag_int': 9,
                    'tag_bool': False,
                },
                data={
                    'baz': bar,
                }
            )
            self.source.insert_times_data(
                measurement='measure3',
                ts=ts,
                tags={
                    'tag_kw': 'tag2',
                    'tag_int': 7,
                    'tag_bool': True,
                },
                data={
                    'baz': -bar,
                }
            )

        self.source.commit()

    def tearDown(self):
        self.source.drop()

    def test_validation(self):
        with self.assertRaises(errors.Invalid):
            InfluxDataSource({
                'addr': 'localhost',
            })
        with self.assertRaises(errors.Invalid):
            InfluxDataSource({
                'database': 'foo',
            })

    def test_build_time_predicates(self):
        self.assertEqual(
            _build_time_predicates(), [],
        )
        self.assertEqual(
            _build_time_predicates(
                from_date=1515404366.1234, to_date="2018-01-08T14:59:25.456Z",
            ),
            [
                "time >= 1515404366123400000",
                "time < 1515423565456000000",
            ],
        )

    def test_build_tags_predicates(self):
        self.assertEqual(
            _build_tags_predicates(), [],
        )
        self.assertEqual(
            _build_tags_predicates([
                {'tag': 'foo', 'value': 'bar'},
                {'tag': 'a "', 'value': 'b \''},
            ]), [
                "\"foo\"='bar'",
                "\"a \\\"\"='b \\''",
            ]
        )

    def test_build_times_queries(self):
        where = "time >= 1515404366123400000 and time < 1515423565456000000"
        queries = list(self.source._build_times_queries(
            self.model,
            from_date=1515404366.1234,
            to_date="2018-01-08T14:59:25.456Z",
        ))
        self.assertEqual(
            queries,
            [
                "select MEAN(\"foo\") as \"avg_foo\" from \"measure1\" "\
                "where {} group by time(3000ms);".format(where),
                "select COUNT(\"bar\") as \"count_bar\" from \"measure2\" "\
                "where {} group by time(3000ms);".format(where),
                "select MEAN(\"baz\") as \"avg_baz\" from \"measure1\" "\
                "where {} and \"mytag\"='myvalue' group by time(3000ms);".format(where),
            ],
        )

        source = InfluxDataSource({
            'name': 'test',
            'addr': ADDR,
            'database': self.db,
            'retention_policy': 'custom',
        })

        queries = list(source._build_times_queries(
            self.model,
            from_date=1515404366.1234,
            to_date="2018-01-08T14:59:25.456Z",
        ))
        from_prefix = '"{}"."custom".'.format(self.db)
        self.assertEqual(
            queries,
            [
                "select MEAN(\"foo\") as \"avg_foo\" from {}\"measure1\" "\
                "where {} group by time(3000ms);".format(from_prefix, where),
                "select COUNT(\"bar\") as \"count_bar\" from {}\"measure2\" "\
                "where {} group by time(3000ms);".format(from_prefix, where),
                "select MEAN(\"baz\") as \"avg_baz\" from {}\"measure1\" "\
                "where {} and \"mytag\"='myvalue' group by time(3000ms);".format(from_prefix, where),
            ],
        )

    def test_get_times_data(self):
        logging.info("[%d %d]", self.t0, self.t0)
        res = self.source.get_times_data(
            self.model,
            from_date=self.t0,
            to_date=self.t0 + 8,
        )

        foo_avg = []
        bar_count = []

        for line in res:
            foo_avg.append(nan_to_none(line[1][0]))
            bar_count.append(nan_to_none(line[1][1]))

        self.assertEqual(foo_avg, [2.5, None, 4.0])
        self.assertEqual(bar_count, [2.0, 0, 1.0])

    def test_match_all(self):
        model = TimeSeriesModel(dict(
            name="test-model",
            offset=30,
            span=300,
            bucket_interval=3,
            interval=60,
            features=FEATURES_MATCH_ALL_TAG1,
            threshold=30,
        ))
        res = self.source.get_times_data(
            model,
            from_date=self.t0,
            to_date=self.t0 + 8,
        )
        baz_avg = []
        for line in res:
            baz_avg.append(line[1][0])

        np.testing.assert_allclose(
            np.array(baz_avg),
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
            features=FEATURES_MATCH_ALL_TAG2,
            threshold=30,
        ))

        res = self.source.get_times_data(
            model,
            from_date=self.t0,
            to_date=self.t0 + 8,
        )
        baz_avg = []
        for line in res:
            baz_avg.append(line[1][0])

        np.testing.assert_allclose(
            np.array(baz_avg),
            np.array([-216.0, np.nan, -18.0]),
            rtol=0,
            atol=0,
        )

class TestInfluxLong(unittest.TestCase):
    def setUp(self):

        self.db = "test-{}".format(int(datetime.datetime.now().timestamp()))
        self.source = InfluxDataSource({
            'name': 'test',
            'addr': ADDR,
            'database': self.db,
        })
        self.source.drop()
        self.source.init()
        self.storage = TempStorage()

        generator = SinEventGenerator(base=3, sigma=0.05)

        self.to_date = datetime.datetime.now().timestamp()
        self.from_date = self.to_date - 3600 * 24 * 7

        for ts in generator.generate_ts(
            self.from_date,
            self.to_date,
            step_ms=60000,
        ):
            self.source.insert_times_data(
                measurement='measure1',
                ts=ts,
                data={
                    'foo': random.lognormvariate(10, 1)
                },
            )
        self.source.commit()

    def test_train(self):
        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=5,
            bucket_interval=20 * 60,
            interval=60,
            features=FEATURES[0:1],
            threshold=30,
            max_evals=1,
        ))

        # Train
        model.train(self.source, from_date=self.from_date, to_date=self.to_date)

        # Check
        self.assertTrue(model.is_trained)


class TestInfluxFingerprints(loudml.test.TestFingerprints):
    def init_source(self):
        self.database = 'test-voip-%d' % self.from_ts
        logging.info("creating database %s", self.database)
        self.source = InfluxDataSource({
            'name': 'test',
            'type': 'influx',
            'addr': ADDR,
            'database': self.database,
        })
        self.source.drop()
        self.source.init()

    def __del__(self):
        self.source.drop()

class TestInfluxTimes(unittest.TestCase):
    def setUp(self):
        this_day = int(datetime.datetime.now(tz=datetime.timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ).timestamp())

        self.database = 'test-times-%d' % this_day
        logging.info("creating database %s", self.database)
        self.source = InfluxDataSource({
            'name': 'test',
            'type': 'influx',
            'addr': ADDR,
            'database': self.database,
        })
        self.source.drop()
        self.source.init()

        # Sin wave. 600s period.
        generator = SinEventGenerator(base=50, amplitude=50, period=600, sigma=0.01)

        # Normal data in range 06-12
        dt = datetime.datetime(2018, 8, 1, 6, 0)
        from_date = dt.replace(tzinfo=timezone.utc).timestamp()
        dt = datetime.datetime(2018, 8, 1, 12, 0)
        to_date = dt.replace(tzinfo=timezone.utc).timestamp()
        for ts, data in self.generate_data(generator, from_date, to_date, step_ms=1000, errors=0):
            self.source.insert_times_data(
                measurement='test_auto',
                ts=ts,
                data=data,
            )
        self.normal_until = to_date


        # Random 20s drops range 12-13
        dt = datetime.datetime(2018, 8, 1, 12, 0)
        from_date = dt.replace(tzinfo=timezone.utc).timestamp()
        dt = datetime.datetime(2018, 8, 1, 13, 0)
        to_date = dt.replace(tzinfo=timezone.utc).timestamp()
        for ts, data in self.generate_data(generator, from_date, to_date, step_ms=1000, errors=0.0001, burst_ms=20000):
            self.source.insert_times_data(
                measurement='test_auto',
                ts=ts,
                data=data,
            )
        
        # Again normal data in range 13-14
        dt = datetime.datetime(2018, 8, 1, 13, 0)
        from_date = dt.replace(tzinfo=timezone.utc).timestamp()
        dt = datetime.datetime(2018, 8, 1, 14, 0)
        to_date = dt.replace(tzinfo=timezone.utc).timestamp()
        for ts, data in self.generate_data(generator, from_date, to_date, step_ms=1000, errors=0):
            self.source.insert_times_data(
                measurement='test_auto',
                ts=ts,
                data=data,
            )

        # Duplicate normal data in range 06-14 and measurement=normal
        dt = datetime.datetime(2018, 8, 1, 6, 0)
        from_date = dt.replace(tzinfo=timezone.utc).timestamp()
        dt = datetime.datetime(2018, 8, 1, 14, 0)
        to_date = dt.replace(tzinfo=timezone.utc).timestamp()
        for ts, data in self.generate_data(generator, from_date, to_date, step_ms=1000, errors=0):
            self.source.insert_times_data(
                measurement='normal',
                ts=ts,
                data=data,
            )

        self.source.commit()

# queries to plot the data in Chronograf
# SELECT count("foo") AS "count_foo" FROM "test-times-*"."autogen"."test_auto" \
#   WHERE time > :dashboardTime: GROUP BY time(10s) FILL(null)
# SELECT last("count_foo") AS "predicted" FROM "test-times-*"."autogen"."prediction_test" \
#   WHERE time > :dashboardTime: GROUP BY time(10s) FILL(null)
#    def tearDown(self):
#        self.source.drop()

    def generate_data(self, generator, from_date, to_date, step_ms=1000, errors=0, burst_ms=0):
        ano = False
        previous_ts = None
        for ts in generator.generate_ts(from_date, to_date, step_ms=step_ms):
            if ano == False and errors > 0:
                val = random.random()
                if val < errors:
                    ano = True
                    total_burst_ms = 0
                    previous_ts = ts

            if ano == True and total_burst_ms < burst_ms:
                total_burst_ms += (ts - previous_ts) * 1000.0
                previous_ts = ts
            else:
                ano = False
                yield ts, {
                    'foo': random.lognormvariate(10, 1),
                }

    def test_loudmld(self):
        model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=8,
            forecast=4,
            bucket_interval=10,
            interval=10,
            features=[
                {
                    'measurement': 'test_auto',
                    'name': 'count_foo',
                    'metric': 'count',
                    'field': 'foo',
                }
            ],
            max_evals=5,
            min_threshold=0,
            max_threshold=0,
        ))
        model2 = TimeSeriesModel(dict(
            name='normal',
            offset=30,
            span=8,
            forecast=4,
            bucket_interval=10,
            interval=10,
            features=[
                {
                    'measurement': 'normal',
                    'name': 'count_foo',
                    'metric': 'count',
                    'field': 'foo',
                }
            ],
            max_evals=1,
        ))

        # Normal data in range 06-12
        dt = datetime.datetime(2018, 8, 1, 6, 0)
        from_date = dt.replace(tzinfo=timezone.utc).timestamp()
        dt = datetime.datetime(2018, 8, 1, 12, 0)
        to_date = dt.replace(tzinfo=timezone.utc).timestamp()

        print("training model")
        model.train(self.source, from_date, to_date)
        print("training done, mse=", model._state['mse'])

        # simulate loudmld loop in range 11h00 - 13h30
        dt = datetime.datetime(2018, 8, 1, 11, 00)
        from_date = dt.replace(tzinfo=timezone.utc).timestamp()
        dt = datetime.datetime(2018, 8, 1, 13, 30)
        to_date = dt.replace(tzinfo=timezone.utc).timestamp()

        normal = []
        data = self.source.get_times_data(
            model2,
            from_date=from_date,
            to_date=to_date,
        )
        for line in data:
            normal.append(line[1])
        normal = np.array(normal)

        prediction = model.predict2(self.source, from_date, to_date, mse_rtol=4)
        model.detect_anomalies(prediction)
        self.source.save_timeseries_prediction(prediction, model)

        self.source.commit()

        self.assertEqual(
            normal.shape, prediction.predicted.shape
        )
        for j, _ in enumerate(normal):
            if prediction.timestamps[j] >= self.normal_until:
                break

            np.testing.assert_allclose(
                prediction.predicted[j],
                normal[j],
                rtol=0.20,
                atol=50,
            )

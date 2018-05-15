import datetime
import logging
import numpy as np
import os
import random
import time
import unittest

logging.getLogger('tensorflow').disabled = True

import loudml.errors as errors
import loudml.test

from loudml.misc import (
    escape_quotes,
    escape_doublequotes,
)

from loudml.influx import (
    _build_queries,
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

    def test_build_queries(self):
        where = "time >= 1515404366123400000 and time < 1515423565456000000"
        queries = list(_build_queries(
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
            foo_avg.append(line[1][0])
            bar_count.append(line[1][1])

        # TODO: rework bucket handling to have the same behavior than in memdatasource

        self.assertEqual(foo_avg, [2.5, 0, 4.0])
        self.assertEqual(bar_count, [2.0, 0, 1.0])


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

        generator = SinEventGenerator(avg=3, sigma=0.05)

        self.to_date = datetime.datetime.now().timestamp()
        self.from_date = self.to_date - 3600 * 24 * 7

        for ts in generator.generate_ts(self.from_date, self.to_date, step=60):
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

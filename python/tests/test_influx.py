import datetime
import logging
import numpy as np
import os
import time
import unittest

import loudml_new.errors as errors

from loudml_new.influx import (
    _build_queries,
    _build_time_predicates,
    InfluxDataSource,
)

logging.getLogger('tensorflow').disabled = True

from loudml_new.times import TimesModel

FEATURES = [
    {
        'name': 'avg_foo',
        'metric': 'avg',
        'measurement': 'measure1',
        'field': 'foo',
        'nan_is_zero': True,
    },
    {
        'name': 'count_bar',
        'metric': 'count',
        'measurement': 'measure2',
        'field': 'bar',
        'nan_is_zero': True,
    },
]

class TestInfluxDataSource(unittest.TestCase):
    def setUp(self):
        if 'INFLUXDB_ADDR' in os.environ:
            addr = os.environ['INFLUXDB_ADDR']
        else:
            addr = 'localhost'

        bucket_interval = 3

        t0 = int(datetime.datetime.now().timestamp())

        # XXX Bucket returned by InfluxDB are aligne on modulo(bucket_interval), that's why
        # timestamp must be aligned for unit tests.
        t0 -= t0 % bucket_interval

        self.t0 = t0

        self.db = 'test-{}'.format(t0)
        logging.info("creating database %s", self.db)
        self.source = InfluxDataSource(addr=addr, db=self.db)
        self.source.delete_db()
        self.source.create_db()

        self.model = TimesModel(
            name="test-model", data=dict(
            db=self.db,
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
        self.source.delete_db()

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
                "time <= 1515423565456000000",
            ],
        )

    def test_build_queries(self):
        where = "time >= 1515404366123400000 and time <= 1515423565456000000"
        queries = list(_build_queries(
            self.model,
            from_date=1515404366.1234,
            to_date="2018-01-08T14:59:25.456Z",
        ))
        self.assertEqual(
            queries,
            [
                "select MEAN(foo) as avg_foo from measure1 "\
                "where {} group by time(3s);".format(where),
                "select COUNT(bar) as count_bar from measure2 "\
                "where {} group by time(3s);".format(where),
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

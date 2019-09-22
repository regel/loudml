import loudml.vendor  # noqa
from loudml.filestorage import TempStorage
from randevents import SinEventGenerator
from loudml.donut import DonutModel
from loudml.model import Model
from loudml.membucket import MemBucket
from loudml.influx import (
    _build_time_predicates,
    _build_tags_predicates,
    InfluxBucket,
)
from loudml.misc import (
    nan_to_none,
    make_ts,
)
import loudml.errors as errors

import copy
import datetime
import logging
import numpy as np
import os
import random
import unittest

logging.getLogger('tensorflow').disabled = True


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

        # XXX Bucket returned by InfluxDB are aligned on
        # modulo(bucket_interval), that's why
        # timestamp must be aligned for unit tests.
        t0 -= t0 % bucket_interval

        self.t0 = t0

        self.db = 'test-{}'.format(t0)
        logging.info("creating database %s", self.db)
        self.source = InfluxBucket({
            'name': 'test',
            'addr': ADDR,
            'database': self.db,
            'measurement': 'nosetests',
        })
        self.source.drop()
        self.source.init()

        self.model = Model(dict(
            name="test-model",
            offset=30,
            span=300,
            bucket_interval=3,
            interval=60,
            features=FEATURES,
        ))

        data = [
            # (foo, bar, timestamp)
            (1, 33, t0 - 1),  # excluded
            (2, 120, t0), (3, 312, t0 + 1),
            # empty
            (4, 18, t0 + 7),
            (5, 78, t0 + 9),  # excluded
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
            InfluxBucket({
                'addr': 'localhost',
            })
        with self.assertRaises(errors.Invalid):
            InfluxBucket({
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
                {'tag': 'int', 'value': 42},
                {'tag': 'bool', 'value': True},
            ]), [
                "\"foo\"='bar'",
                "\"a \\\"\"='b \\''",
                "(\"int\"='42' OR \"int\"=42)",
                "(\"bool\"='True' OR \"bool\"=True)",
            ]
        )

    def test_build_times_queries(self):
        where = "time >= 1515404366123400000 and time < 1515423565456000000"
        queries = list(self.source._build_times_queries(
            bucket_interval=self.model.bucket_interval,
            features=self.model.features,
            from_date=1515404366.1234,
            to_date="2018-01-08T14:59:25.456Z",
        ))
        self.assertEqual(
            queries,
            [
                "select MEAN(\"foo\") as \"avg_foo\" from \"measure1\" "
                "where {} group by time(3000ms);".format(where),
                "select COUNT(\"bar\") as \"count_bar\" from \"measure2\" "
                "where {} group by time(3000ms);".format(where),
                "select MEAN(\"baz\") as \"avg_baz\" from \"measure1\" "
                "where {} and \"mytag\"='myvalue' group by time(3000ms);".format(
                    where),
            ],
        )

        source = InfluxBucket({
            'name': 'test',
            'addr': ADDR,
            'database': self.db,
            'retention_policy': 'custom',
            'measurement': 'nosetests',
        })

        queries = list(source._build_times_queries(
            bucket_interval=self.model.bucket_interval,
            features=self.model.features,
            from_date=1515404366.1234,
            to_date="2018-01-08T14:59:25.456Z",
        ))
        from_prefix = '"{}"."custom".'.format(self.db)
        self.assertEqual(
            queries,
            [
                "select MEAN(\"foo\") as \"avg_foo\" from {}\"measure1\" "
                "where {} group by time(3000ms);".format(from_prefix, where),
                "select COUNT(\"bar\") as \"count_bar\" from {}\"measure2\" "
                "where {} group by time(3000ms);".format(from_prefix, where),
                "select MEAN(\"baz\") as \"avg_baz\" from {}\"measure1\" "
                "where {} and \"mytag\"='myvalue' group by time(3000ms);".format(
                    from_prefix, where),
            ],
        )

    def test_get_times_data(self):
        logging.info("[%d %d]", self.t0, self.t0)
        res = self.source.get_times_data(
            bucket_interval=self.model.bucket_interval,
            features=self.model.features,
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

    def test_get_times_data2(self):
        res = self.source.get_times_data(
            bucket_interval=self.model.bucket_interval,
            features=self.model.features,
            from_date=self.t0,
            to_date=self.t0 + 8,
        )

        # _source to write aggregate data to RAM
        _source = MemBucket()
        _features = copy.deepcopy(self.model.features)
        for _, feature in enumerate(self.model.features):
            feature.metric = 'avg'

        i = None
        for i, (_, val, timeval) in enumerate(res):
            bucket = {
                feature.field: val[i]
                for i, feature in enumerate(self.model.features)
            }
            bucket.update({'timestamp': make_ts(timeval)})
            _source.insert_times_data(bucket)

        res2 = _source.get_times_data(
            bucket_interval=self.model.bucket_interval,
            features=self.model.features,
            from_date=self.t0,
            to_date=self.t0 + 8,
        )
        self.model.features = _features

        for i, (_, val2, timeval2) in enumerate(res2):
            (_, val, timeval) = res[i]
            np.testing.assert_allclose(val, val2)

    def test_match_all(self):
        model = Model(dict(
            name="test-model",
            offset=30,
            span=300,
            bucket_interval=3,
            interval=60,
            features=FEATURES_MATCH_ALL_TAG1,
        ))
        res = self.source.get_times_data(
            bucket_interval=model.bucket_interval,
            features=model.features,
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

        model = Model(dict(
            name="test-model",
            offset=30,
            span=300,
            bucket_interval=3,
            interval=60,
            features=FEATURES_MATCH_ALL_TAG2,
        ))

        res = self.source.get_times_data(
            bucket_interval=model.bucket_interval,
            features=model.features,
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
        self.source = InfluxBucket({
            'name': 'test',
            'addr': ADDR,
            'database': self.db,
            'measurement': 'nosetests',
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
        model = DonutModel(dict(
            name='test',
            offset=30,
            span=5,
            bucket_interval=20 * 60,
            interval=60,
            features=FEATURES[0:1],
            max_evals=1,
        ))

        # Train
        model.train(self.source, from_date=self.from_date,
                    to_date=self.to_date)

        # Check
        self.assertTrue(model.is_trained)

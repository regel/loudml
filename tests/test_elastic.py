from loudml.donut import (
    TimeSeriesPrediction,
)
from loudml.model import Model
from loudml.elastic import ElasticsearchBucket
import loudml.bucket
import loudml.config

import copy
import datetime
import logging
import numpy as np
import os
import time
import unittest

logging.getLogger('tensorflow').disabled = True


FEATURES = [
    {
        'name': 'avg_foo',
        'metric': 'avg',
        'field': 'foo',
        'default': 0,
    },
]

FEATURES_MATCH_ALL_TAG1 = [
    {
        'name': 'avg_baz',
        'metric': 'avg',
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
        'field': 'baz',
        'match_all': [
            {'tag': 'tag_int', 'value': 7},
            {'tag': 'tag_bool', 'value': True},
        ],
    },
]


class TestElasticBucket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bucket_interval = 3

        t0 = int(datetime.datetime.now().timestamp())
        t0 -= t0 % bucket_interval
        cls.t0 = t0

        cls.index = "test-{}".format(t0)
        cls.sink_index = "test-{}-prediction".format(t0)

        logging.info("creating index %s", cls.index)
        if os.environ.get('ELASTICSEARCH_ADDR', None) is None:
            # tip: useful tool to query ES AWS remotely:
            # npm install aws-es-curl -g
            settings = dict(
                name='aws',
                type='elasticsearch_aws',
                doc_type='doc',
                host=os.environ['ELASTICSEARCH_HOST'],
                region='eu-west-1',
                get_boto_credentials=False,
                access_key=os.environ['AWS_ACCESS_KEY_ID'],
                secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            )

            settings['index'] = cls.index
            cls.source = loudml.bucket.load_bucket(settings)

            settings = copy.deepcopy(settings)
            settings['index'] = cls.sink_index
            cls.sink = loudml.bucket.load_bucket(settings)
        else:
            settings = {
                'name': 'test',
                'addr': os.environ['ELASTICSEARCH_ADDR'],
                'index': cls.index,
                'doc_type': 'nosetests',
            }
            cls.source = ElasticsearchBucket(settings)

            settings = copy.deepcopy(settings)
            settings['index'] = cls.sink_index
            cls.sink = ElasticsearchBucket(settings)

        data_schema = {
            "foo": {"type": "integer"},
            "bar": {"type": "integer"},
            "baz": {"type": "integer"},
            "tag_kw": {"type": "keyword"},
            "tag_int": {"type": "integer"},
            "tag_bool": {"type": "boolean"},
        }
        cls.source.drop()
        cls.source.init(data_schema=data_schema)

        cls.model = Model(dict(
            name='times-model',  # not test-model due to TEMPLATE
            offset=30,
            span=300,
            bucket_interval=bucket_interval,
            interval=60,
            features=FEATURES,
            threshold=30,
        ))

        data = [
            # (foo, bar|baz, timestamp)
            (1, 33, t0 - 1),  # excluded
            (2, 120, t0), (3, 312, t0 + 1),
            # empty
            (4, 18, t0 + 7),
            (5, 78, t0 + 9),  # excluded
        ]
        for foo, bar, ts in data:
            cls.source.insert_times_data(
                ts=ts,
                data={
                    'foo': foo,
                }
            )
            cls.source.insert_times_data(
                ts=ts,
                data={
                    'bar': bar,
                }
            )
            cls.source.insert_times_data(
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
            cls.source.insert_times_data(
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

        cls.source.commit()

        # Let elasticsearch indexes the data before querying it
        time.sleep(10)

    @classmethod
    def tearDownClass(cls):
        cls.sink.drop()
        cls.source.drop()

    def test_get_index_name(self):
        ts = 1527156069

        self.assertEqual(self.source.get_index_name(), self.index)
        self.assertEqual(self.source.get_index_name("test"), "test")
        self.assertEqual(
            self.source.get_index_name("test", timestamp=ts),
            "test"
        )
        self.assertEqual(
            self.source.get_index_name("test-*", timestamp=ts),
            "test-2018.05.24",
        )

    def test_get_times_data(self):
        res = self.source.get_times_data(
            bucket_interval=self.model.bucket_interval,
            features=self.model.features,
            from_date=self.t0,
            to_date=self.t0 + 8,
        )

        foo_avg = []
        for line in res:
            foo_avg.append(line[1][0])

        np.testing.assert_allclose(
            np.array(foo_avg),
            np.array([2.5, np.nan, 4.0]),
            rtol=0,
            atol=0,
        )

    def test_save_timeseries_prediction(self):
        now_ts = datetime.datetime.now().timestamp()

        timestamps = [
            now_ts,
            now_ts + self.model.bucket_interval,
        ]
        predicted = [4.0, 2.0]
        observed = [4.1, 1.9]

        prediction = TimeSeriesPrediction(
            self.model,
            timestamps=timestamps,
            predicted=np.array(predicted),
            observed=np.array(observed),
        )

        self.sink.init(data_schema=prediction.get_schema())
        self.sink.save_timeseries_prediction(
            prediction, tags=self.model.get_tags())
        self.sink.refresh()

        res = self.sink.search(
            routing=self.model.routing,
            size=100,
            body={}
        )

        hits = res['hits']['hits']
        self.assertEqual(len(hits), 2)

        for i, hit in enumerate(sorted(
            hits, key=lambda x: x['_source']['timestamp'])
        ):
            source = hit['_source']
            self.assertEqual(source, {
                'avg_foo': predicted[i],
                '@avg_foo': observed[i],
                'timestamp': int(timestamps[i] * 1000),
                'model': self.model.name,
            })

    def test_match_all(self):
        model = Model(dict(
            name="times-model",
            offset=30,
            span=300,
            bucket_interval=3,
            interval=60,
            features=FEATURES_MATCH_ALL_TAG1,
            threshold=30,
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
            name="times-model",
            offset=30,
            span=300,
            bucket_interval=3,
            interval=60,
            features=FEATURES_MATCH_ALL_TAG2,
            threshold=30,
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

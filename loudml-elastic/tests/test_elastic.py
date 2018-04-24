import datetime
import logging
import numpy as np
import os
import time
import unittest

logging.getLogger('tensorflow').disabled = True

import loudml.errors as errors
import loudml.test

from loudml.elastic import ElasticsearchDataSource

from loudml.timeseries import TimeSeriesModel

TEMPLATE = {
    "template": "test-*",
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "codec":"best_compression"
    },
    "mappings": {
        "generic": {
            "include_in_all": True,
            "properties": {
                "timestamp": {
                    "type": "date"
                },
                "foo": {
                    "type": "integer"
                },
            },
        },
    },
}

FEATURES = [
    {
        'name': 'avg_foo',
        'metric': 'avg',
        'field': 'foo',
        'default': 0,
    },
]

class TestElasticDataSource(unittest.TestCase):
    def setUp(self):
        bucket_interval = 3

        t0 = int(datetime.datetime.now().timestamp())
        t0 -= t0 % bucket_interval
        self.t0 = t0

        self.index = 'test-%d' % t0
        logging.info("creating index %s", self.index)
        self.source = ElasticsearchDataSource({
            'name': 'test',
            'addr': os.environ['ELASTICSEARCH_ADDR'],
            'index': self.index,
        })
        self.source.delete_index()
        self.source.create_index(TEMPLATE)

        self.model = TimeSeriesModel(dict(
            name='test',
            offset=30,
            span=300,
            bucket_interval=bucket_interval,
            interval=60,
            features=FEATURES,
            threshold=30,
        ))

        data = [
            # (foo, timestamp)
            (1, t0 - 1), # excluded
            (2, t0), (3, t0 + 1),
            # empty
            (4, t0 + 7),
            (5, t0 + 9), # excluded
        ]
        for entry in data:
            self.source.insert_times_data(
                ts=entry[1],
                data={
                    'foo': entry[0],
                },
            )
        self.source.commit()

        # Let elasticsearch indexes the data before querying it
        time.sleep(10)

    def tearDown(self):
        self.source.delete_index()

    def test_get_times_data(self):
        res = self.source.get_times_data(
            self.model,
            from_date=self.t0,
            to_date=self.t0 + 8,
        )

        self.assertEqual(
            [line[1] for line in res],
            [[2.5], [0.0], [4.0]],
        )

VOIP_TEMPLATE = {
    "template": "test-voip-*",
    "mappings": {
        "session": {
            "properties": {
                "@timestamp": {
                    "type": "date"
                },
                "duration": {
                    "type": "integer"
                },
                "caller": {
                    "type": "keyword"
                },
                "international": {
                    "type": "boolean"
                },
                "toll_call": {
                    "type": "boolean"
                }
            }
        }
    }
}

class TestElasticFingerprints(loudml.test.TestFingerprints):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Let elasticsearch indexes the data before querying it
        time.sleep(10)

    def init_source(self):
        addr = os.environ.get('ELASTICSEARCH_ADDR', 'localhost:9200')
        self.index = 'test-voip-{}'.format(self.from_ts)
        logging.info("creating index %s", self.index)
        self.source = ElasticsearchDataSource({
            'name': 'test',
            'type': 'elasticsearch',
            'addr': addr,
            'index': self.index,
        })
        self.source.delete_index()
        self.source.create_index(VOIP_TEMPLATE)

    def __del__(self):
        self.source.delete_index()

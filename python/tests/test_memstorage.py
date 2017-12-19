import datetime
import numpy as np
import os
import unittest

import loudml_new.errors as errors
from loudml_new.times import TimesModel
from loudml_new.memstorage import MemStorage

FEATURES = [
    {
        'name': 'avg_foo',
        'metric': 'avg',
        'field': 'foo',
        'nan_is_zero': True,
    },
]

def get_ms_ts():
    return int(datetime.datetime.now().timestamp() * 1000)

class TestMemStorage(unittest.TestCase):
    def setUp(self):
        self.storage = MemStorage()
        self.model = TimesModel("test", dict(
            index='test',
            offset=30,
            span=300,
            bucket_interval=3,
            interval=60,
            features=FEATURES,
            threshold=30,
        ))
        data = [
            (1, 0), # excluded
            (2, 1), (3, 2),
            # empty
            (4, 8),
            (5, 10), # excluded
        ]
        for entry in data:
            self.storage.insert_times_data('test', {
                'foo': entry[0],
                'timestamp': entry[1],
            })


    def tearDown(self):
        del self.storage

    def test_get_times_buckets(self):
        res = self.storage.get_times_buckets('test',
            from_date=1,
            to_date=9,
            bucket_interval=3,
        )
        self.assertEqual(
            [[entry.data['foo'] for entry in bucket.data] for bucket in res],
            [[2, 3], [], [4]],
        )

    def test_get_times_data(self):
        res = self.storage.get_times_data(self.model, from_date=1, to_date=9)

        self.assertEqual(
            [line[1] for line in res],
            [[2.5], [0], [4.0]],
        )

import datetime
import logging
import os
import random
import unittest

logging.getLogger('tensorflow').disabled = True

from loudml_new.randevents import EventGenerator
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

class TestTimes(unittest.TestCase):
    def setUp(self):
        self.storage = MemStorage()

        generator = EventGenerator(lo=2, hi=4, sigma=0.05)

        self.to_date = datetime.datetime.now().timestamp()
        self.from_date = self.to_date - 3600 * 24 * 7

        for ts in generator.generate_ts(self.from_date, self.to_date, step=60):
            self.storage.insert_times_data('test', {
                'timestamp': ts,
                'foo': random.lognormvariate(10, 1)
            })

    def test_train(self):
        self.assertGreater(len(self.storage.data['test']), 0)

        model = TimesModel('test', dict(
            index='test',
            offset=30,
            span=24 * 3600,
            bucket_interval=20 * 60,
            interval=60,
            features=FEATURES,
            threshold=30,
            max_evals=1,
        ))

        # Train
        model.train(self.storage)

        # Check
        self.assertTrue(model.is_trained)

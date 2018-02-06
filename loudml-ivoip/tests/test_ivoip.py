import datetime
import logging
import os
import random
import time
import unittest

import numpy as np

from loudml.ivoip import (
    IVoipDataSource,
    IVoipFingerprintsModel,
)

TEMPLATE = {
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

class TestIVoipFingerprints(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from_ts = int(datetime.datetime.now(tz=datetime.timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ).timestamp())
        self.from_ts = from_ts
        self.to_ts = from_ts + 3600 * 24

        self.index = 'test-voip-%d' % from_ts
        logging.info("creating index %s", self.index)
        self.source = IVoipDataSource({
            'name': 'test',
            'addr': os.environ['ELASTICSEARCH_ADDR'],
            'index': self.index,
        })
        self.source.delete_index()
        self.source.create_index(TEMPLATE)

        self.model = IVoipFingerprintsModel(dict(
            name='test',
            key='caller',
            max_keys=1024,
            height=50,
            width=50,
            interval=60,
            span="24h",
        ))

        self.callers = [
            "33612345678",
            "33601020304",
            "33688774455",
        ]

        def add(i, v):
            d, i, p = v
            self.source.insert_times_data(
                ts=self.from_ts + i * step,
                data={
                    'caller': caller,
                    'duration': d,
                    'international': i,
                    'toll_call': p,
                },
                timestamp_field=self.model.timestamp_field,
            )

        # Profile 0: very busy, short calls, low variation
        caller = self.callers[0]
        data = [
            (30, False, False),
            (60, False, False),
            (90, False, False),
            (40, False, False),
            (120, False, False),
            (60, True, False),
            (30, False, False),
            (150, False, True),
            (60, False, False),
            (80, False, False),
        ]
        step = int((self.to_ts - self.from_ts) / len(data))
        for i, v in enumerate(data):
            add(i, v)

        # Profile 1: less busy, high variation
        caller = self.callers[1]
        data = [
            (5, False, False),
            (1200, False, False),
            (800, False, False),
            (20, False, False),
            (500, False, False),
            (760, True, False),
            (60, False, False),
        ]
        step = int((self.to_ts - self.from_ts) / len(data))
        for i, v in enumerate(data):
            add(i, v)

        # Profile 2: lazy, low variation, international
        caller = self.callers[2]
        data = [
            (5, True, False),
            (1200, True, False),
            (800, True, False),
            (20, True, True),
        ]
        step = int((self.to_ts - self.from_ts) / len(data))
        for i, v in enumerate(data):
            add(i, v)

        # Insert data
        self.source.commit()

        # Let elasticsearch indexes the data before querying it
        time.sleep(10)

    def __del__(self):
        self.source.delete_index()

    def test_model(self):
        self.assertEqual(self.model.type, 'ivoip_fingerprints')

    def test_get_quadrant_data(self):
        res = self.source.get_quadrant_data(
            self.model,
            from_date=self.from_ts,
            to_date=self.to_ts,
        )

        res = list(res)
        self.assertEqual(len(res), 3)
        # TODO more checks needed

    def _require_training(self):
        if self.model.is_trained:
            return
        self.model.train(self.source, self.from_ts, self.to_ts)

    def test_train(self):
        self._require_training()
        self.assertTrue(self.model.is_trained)

    def test_predict(self):
        self._require_training()
        prediction = self.model.predict(
            self.source,
            self.from_ts,
            self.to_ts,
        )
        self.assertEqual(len(prediction.fingerprints), 3)

        # For now, only the data format is checked
        for fp in prediction.fingerprints:
            self.assertTrue(fp['key'] in self.callers)

            x, y = fp['location']
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLess(x, self.model.w)
            self.assertLess(y, self.model.h)

            self.assertEqual(len(fp['time_range']), 2)

            self.assertEqual(len(fp['_fingerprint']), 36)
            self.assertEqual(len(fp['fingerprint']), 36)

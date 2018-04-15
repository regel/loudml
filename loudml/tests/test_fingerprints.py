import datetime
import logging
import os
import random
import time
import unittest

import numpy as np

from loudml.fingerprints import (
    FingerprintsModel,
    FingerprintsPrediction,
)

logging.getLogger('tensorflow').disabled = True

from loudml.elastic import ElasticsearchDataSource

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

class TestFingerprints(unittest.TestCase):
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
        self.source = ElasticsearchDataSource({
            'name': 'test',
            'type': 'elasticsearch',
            'addr': os.environ['ELASTICSEARCH_ADDR'],
            'index': self.index,
        })
        self.source.delete_index()
        self.source.create_index(TEMPLATE)

        self.model = FingerprintsModel(dict(
            name='test',
            key='caller',
            max_keys=1024,
            height=50,
            width=50,
            interval=60,
            span="24h",
            timestamp_field="@timestamp",
            use_daytime=True,
            daytime_interval="6h",
            offset="30s",
            aggregations=
              [dict(
                measurement="xdr",
                features=[
                  dict(
                    field="duration",
                    name="count-all",
                    metric="count"
                  ),
                  dict(
                    field="duration",
                    name="mean-all-duration",
                    metric="avg"
                  ),
                  dict(
                    field="duration",
                    name="std-all-duration",
                    metric="stddev"
                  )
                ]
              ),
              dict(
                measurement="xdr",
                match_all=[dict(tag="international", value="true")],
                features=[
                  dict(
                    field="duration",
                    name="count-international",
                    metric="count"
                  ),
                  dict(
                    field="duration",
                    name="mean-international-duration",
                    metric="avg"
                  ),
                  dict(
                    field="duration",
                    name="std-international-duration",
                    metric="stddev"
                  )
                ]
              ),
              dict(
                measurement="xdr",
                match_all=[dict(tag="toll_call", value="true")],
                features=[
                  dict(
                    field="duration",
                    name="count-premium",
                    metric="count"
                  ),
                  dict(
                    field="duration",
                    name="mean-premium-duration",
                    metric="avg"
                  ),
                  dict(
                    field="duration",
                    name="std-premium-duration",
                    metric="stddev"
                  )
                ]
              )],
        ))

        self.callers = [
            "33612345678",
            "33601020304",
            "33688774455",
        ]

        self.expected_fp={}

        self.expected_fp['33601020304'] = [2.0, 602.5, 597.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 410.0, 390.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 630.0, 130.0, 1.0, 760.0, 0.0, 0.0, 0.0, 0.0, 1.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.expected_fp['33612345678'] = [3.0, 60.0, 24.49489742783178, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 80.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 80.0, 50.99019513592785, 1.0, 60.0, 0.0, 1.0, 150.0, 0.0, 2.0, 70.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.expected_fp['33688774455'] = [1.0, 5.0, 0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1200.0, 0.0, 1.0, 1200.0, 0.0, 0.0, 0.0, 0.0, 1.0, 800.0, 0.0, 1.0, 800.0, 0.0, 0.0, 0.0, 0.0, 1.0, 20.0, 0.0, 1.0, 20.0, 0.0, 1.0, 20.0, 0.0]


        def add(j, v):
            d, i, p = v
            tags=dict()
            tags['caller']=caller
            tags['international']=i
            tags['toll_call']=p

            self.source.insert_times_data(
                ts=self.from_ts + j * step,
                data={
                    'duration': d,
                },
                measurement='xdr',
                tags=tags,
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
        self.assertEqual(self.model.type, 'fingerprints')

    def test_get_quadrant_data(self):
        res = self.source.get_quadrant_data(
            self.model,
            self.model.aggs[0],
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

            self.assertEqual(fp['fingerprint'], self.expected_fp[fp['key']])


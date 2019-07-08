import loudml.vendor  # noqa: F401

from loudml.server import app
from loudml.server import g_app_init
from loudml.server import g_app_stop

from loudml.influx import (
    InfluxDataSource,
)

from loudml.randevents import SinEventGenerator

import unittest

import datetime
import logging
import os
import random
import math
import json
import tempfile
import shutil
import time

logging.getLogger('tensorflow').disabled = True


FEATURE_COUNT_FOO = {
    'measurement': 'bar',
    'name': 'count_foo',
    'metric': 'count',
    'field': 'foo',
    'default': 0,
}

FEATURE_AVG_FOO = {
    'measurement': 'bar',
    'name': 'avg_foo',
    'metric': 'avg',
    'field': 'foo',
    'default': 10,
}

FEATURES = [FEATURE_COUNT_FOO]

CONFIG = """
---
datasources:
 - name: nose
   type: influxdb
   addr: {}
   database: {}
   create_database: true
   retention_policy: autogen
   max_series_per_request: 2000

storage:
  path: {}

server:
  listen: localhost:8077
"""

if 'INFLUXDB_ADDR' in os.environ:
    ADDR = os.environ['INFLUXDB_ADDR']
else:
    ADDR = 'localhost'


def read_job_id(res):
    job_id = res.data.decode('utf-8').strip('\n').replace('\"', '')
    return job_id


def state_is_done(state):
    return state in ['done', 'failed', 'canceled']


class AppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        self.bucket_interval = 20 * 60

        t0 = int(datetime.datetime.now().timestamp())
        t0 -= t0 % self.bucket_interval
        self.t0 = t0

        self.db = 'test-{}'.format(t0)
        logging.info("creating database %s", self.db)
        self.source = InfluxDataSource({
            'name': 'nose',
            'addr': ADDR,
            'database': self.db,
        })
        self.source.drop()
        self.source.init()

        self.generator = SinEventGenerator(base=3, amplitude=3, sigma=0.01)

        to_date = datetime.datetime.now().timestamp()

        # Be sure that date range is aligned
        self.to_date = math.floor(to_date / self.bucket_interval) * self.bucket_interval  # noqa E501
        self.from_date = self.to_date - 3600 * 24 * 7 * 3

        for ts in self.generator.generate_ts(self.from_date, self.to_date, step_ms=600000):  # noqa E501
            self.source.insert_times_data(
                measurement='bar',
                ts=ts,
                data={
                    'foo': random.normalvariate(10, 1),
                }
            )

        self.source.commit()

        self.dirpath = tempfile.mkdtemp()
        configyml = os.path.join(self.dirpath, 'config.yml')
        cfg = open(configyml, 'w')
        cfg.write(CONFIG.format(ADDR, self.db, self.dirpath))
        cfg.close()

        g_app_init(configyml)

    def tearDown(self):
        g_app_stop()
        self.source.drop()
        shutil.rmtree(self.dirpath)

    def _wait_job(self, job_id):
        state = None
        while not state_is_done(state):
            time.sleep(5)
            result = self.app.get(
                '/jobs/{}'.format(job_id)
            )
            res = json.loads(result.data.decode('utf-8'))
            print(res)
            state = res['state']
        return state

    def _get_models(self):
        result = self.app.get(
            '/models',
        )
        self.assertEqual(result.status_code, 200)
        d = json.loads(result.data.decode('utf-8'))
        return d

    def _require_model(self):
        d = self._get_models()
        if len(d) > 0 and d[0]['name'] == 'test-model':
            return

        model = dict(
            name='test-model',
            default_datasource='nose',
            offset=30,
            span=24 * 3,
            bucket_interval=self.bucket_interval,
            interval=60,
            features=FEATURES,
            grace_period="140m",  # = 7 points
            max_threshold=99.7,
            min_threshold=68,
            max_evals=1,
        )
        model['type'] = 'donut'

        result = self.app.put(
            '/models',
            follow_redirects=True,
            content_type='application/json',
            data=json.dumps(model),
        )
        self.assertEqual(result.status_code, 201)
        d = self._get_models()
        self.assertEqual(len(d), 1)
        self.assertEqual(d[0]['settings']['name'], 'test-model')
        return d[0]

    def _require_training(self):
        model = self._require_model()
        if model['state']['trained']:
            return
        result = self.app.post(
            '/models/{}/_train?from={}&to={}'.format(
                'test-model',
                str(self.from_date),
                str(self.to_date),
            ),
        )
        job_id = read_job_id(result)
        status = self._wait_job(job_id)
        self.assertEqual(status, 'done')

    def test_train(self):
        self._require_training()

    def test_home(self):
        result = self.app.get('/')
        self.assertEqual(result.status_code, 200)
        d = json.loads(result.data.decode('utf-8'))
        self.assertIsNotNone(d.get('host_id'))
        self.assertIsNotNone(d.get('version'))

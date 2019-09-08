import loudml.vendor  # noqa: F401

from loudml.influx import (
    InfluxDataSource,
)

from loudml.randevents import SinEventGenerator

import unittest
import requests

import datetime
import logging
import os
import random
import math
import json
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


def read_job_id(res):
    job_id = res.strip('\n').replace('\"', '')
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
        self._jwt = None

    def setUp(self):
        if 'LOUDML_ADDR' in os.environ:
            self.loudml_addr = os.environ['LOUDML_ADDR']
        else:
            self.loudml_addr = 'localhost:8077'

        self.bucket_interval = 20 * 60

        t0 = int(datetime.datetime.now().timestamp())
        t0 -= t0 % self.bucket_interval
        self.t0 = t0

        self.db = 'test-{}'.format(t0)
        logging.info("creating database %s", self.db)
        if 'INFLUXDB_ADDR' in os.environ:
            addr = os.environ['INFLUXDB_ADDR']
        else:
            addr = 'localhost'

        self.source = InfluxDataSource({
            'name': 'nosetests',
            'addr': addr,
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

    def tearDown(self):
        self.source.drop()

    def get(self, url, data=None, **kwargs):
        headers = kwargs.pop('headers', {})
        if self._jwt:
            headers['Authorization'] = 'Bearer {}'.format(self._jwt)
        return requests.get(
            self.get_url(url),
            data=data,
            headers=headers,
            **kwargs
        )

    def post(self, url, data=None, **kwargs):
        headers = kwargs.pop('headers', {})
        if self._jwt:
            headers['Authorization'] = 'Bearer {}'.format(self._jwt)
        return requests.post(
            self.get_url(url),
            data=data,
            headers=headers,
            **kwargs
        )

    def patch(self, url, data=None, **kwargs):
        headers = kwargs.pop('headers', {})
        if self._jwt:
            headers['Authorization'] = 'Bearer {}'.format(self._jwt)
        return requests.patch(
            self.get_url(url),
            data=data,
            headers=headers,
            **kwargs
        )

    def delete(self, url, data=None, **kwargs):
        headers = kwargs.pop('headers', {})
        if self._jwt:
            headers['Authorization'] = 'Bearer {}'.format(self._jwt)
        return requests.delete(
            self.get_url(url),
            data=data,
            headers=headers,
            **kwargs
        )

    def put(self, url, data=None, content_type=None, **kwargs):
        headers = kwargs.pop('headers', {})
        if self._jwt:
            headers['Authorization'] = 'Bearer {}'.format(self._jwt)
        if content_type:
            headers['Content-Type'] = content_type
        return requests.put(
            self.get_url(url),
            data=data,
            headers=headers,
            **kwargs
        )

    def get_url(self, url):
        if 'USE_SSL' in os.environ:
            scheme = 'https://'
        else:
            scheme = 'http://'

        return scheme + self.loudml_addr + url

    def _wait_job(self, job_id):
        state = None
        while not state_is_done(state):
            time.sleep(5)
            response = self.get(
                '/jobs/{}'.format(job_id)
            )
            res = response.json()
            # print(res)
            state = res['state']
        return state

    def _get_models(self):
        response = self.get(
            '/models',
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def _get_sources(self):
        response = self.get(
            '/datasources',
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def _require_source(self):
        sources = {
            source['name']: source
            for source in self._get_sources()
        }
        if self.db in sources:
            return

        if 'INFLUXDB_ADDR' in os.environ:
            addr = os.environ['INFLUXDB_ADDR']
        else:
            addr = 'localhost:8086'

        source = {
            'name': self.db,
            'type': 'influxdb',
            'addr': addr,
            'database': self.db,
            'create_database': 'true',
            'retention_policy': 'autogen',
            'max_series_per_request': 2000,
        }
        response = self.put(
            '/datasources',
            content_type='application/json',
            data=json.dumps(source),
        )
        self.assertEqual(response.status_code, 201)
        sources = {
            source['name']: source
            for source in self._get_sources()
        }
        self.assertTrue(self.db in sources)

    def _del_model(self, model_name):
        response = self.delete(
            '/models/{}'.format(model_name),
        )
        self.assertTrue(response.status_code in [200, 404])

    def _require_model(self):
        self._require_source()
        models = {
            model['settings']['name']: model
            for model in self._get_models()
        }
        if 'test-model' in models:
            return models['test-model']

        model = dict(
            name='test-model',
            default_datasource=self.db,
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

        response = self.put(
            '/models',
            content_type='application/json',
            data=json.dumps(model),
        )
        self.assertEqual(response.status_code, 201)
        models = {
            model['settings']['name']: model
            for model in self._get_models()
        }
        self.assertTrue('test-model' in models)
        return models['test-model']

    def _require_training(self):
        model = self._require_model()
        if model['state']['trained']:
            return
        response = self.post(
            '/models/{}/_train?from={}&to={}'.format(
                'test-model',
                str(self.from_date),
                str(self.to_date),
            ),
        )
        job_id = read_job_id(response.text)
        status = self._wait_job(job_id)
        self.assertEqual(status, 'done')

    def test_training(self):
        self._del_model('test-model')
        self._require_training()

    def test_home(self):
        response = self.get('/')
        self.assertEqual(response.status_code, 200)
        home = response.json()
        self.assertIsNotNone(home.get('host_id'))
        self.assertIsNotNone(home.get('version'))

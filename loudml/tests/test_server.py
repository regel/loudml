import os
import unittest

from loudml import server
from loudml import config


class TestServer(unittest.TestCase):
    def setUp(self):
        server.app.config['TESTING'] = True
        server.g_config = config.load_config(
            os.path.join(
                os.path.dirname(__file__),
                '..',
                'examples',
                'config.yml'))
        self.client = server.app.test_client()

    def test_route_slash(self):
        rv = self.client.get('/')
        self.assertTrue(rv.is_json)
        data = rv.get_json()
        self.assertIn('tagline', data)

    def test_route_license(self):
        rv = self.client.get('/license')
        self.assertTrue(rv.is_json)
        data = rv.get_json()
        self.assertIn('features', data)
        self.assertIn('nrmodels', data['features'])

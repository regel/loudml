import os
import unittest
from unittest import mock

from loudml import server
from loudml import config


def mocked_get_distribution(*args, **kwargs):
    class distribution:
        version = '1.5'

    return distribution()


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

    @mock.patch('pkg_resources.get_distribution',
                side_effect=mocked_get_distribution)
    def test_route_slash(self, mock_get):
        rv = self.client.get('/')
        self.assertTrue(rv.is_json)
        data = rv.get_json()
        self.assertIn('tagline', data)

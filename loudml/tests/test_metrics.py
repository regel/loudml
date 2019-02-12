import unittest
from unittest import mock
from unittest.mock import MagicMock

from loudml.metrics import send_metrics
from loudml.metrics import MyConfigParser
from loudml.dummystorage import DummyStorage


def mocked_get_distribution(*args, **kwargs):
    class distribution:
        version = '1.5'

    return distribution()


def mocked_requests_post(*args, **kwargs):
    pass


class TestMetrics(unittest.TestCase):
    @mock.patch('pkg_resources.get_distribution',
                side_effect=mocked_get_distribution)
    @mock.patch('requests.post', side_effect=mocked_requests_post)
    def test_send_metrics(self, mock_get, mock_get2):
        config = {'enable': True}
        storage = DummyStorage()
        storage.list_models = MagicMock(return_value=['foo', 'bar'])
        MyConfigParser.read = MagicMock()
        MyConfigParser.get = MagicMock(return_value='CentOS')

        send_metrics(config, storage)

        storage.list_models.assert_called()

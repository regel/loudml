import unittest
from unittest import mock

from loudml.metrics import send_metrics


def mocked_requests_post(*args, **kwargs):
    pass


class TestMetrics(unittest.TestCase):
    @mock.patch('requests.post', side_effect=mocked_requests_post)
    def test_send(self, mock_get):
        send_metrics()

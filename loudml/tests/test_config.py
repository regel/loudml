import unittest

from loudml.config import Config


class TestConfig(unittest.TestCase):
    def test_default_config(self):
        c = Config({})
        self.assertTrue(c.metrics['enable'])

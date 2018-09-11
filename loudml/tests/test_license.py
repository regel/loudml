import os
import unittest

import loudml

from loudml import errors
from loudml.config import Config
from loudml.license import License
from loudml.model import load_model


class TestLicense(unittest.TestCase):
    # Test license.
    # Version: 1
    # Integrity: OK
    # Contents:
    #   {
    #      "features": {"nrmodels": 50, "fingerprints": true},
    #      "version": 1,
    #      "exp_date": "2017-01-01",
    #      "hostid": "any",
    #      "serial_num": "0"
    #    }
    def setUp(self):
        lic = License()
        lic.load(os.path.join(
            os.path.dirname(__file__),
            'resources',
            'license1.lic'
        ))
        self.license1 = lic

    def test_expired(self):
        self.assertTrue(self.license1.has_expired())

    def test_model_not_allowed(self):
        config = Config({})
        settings = {'type': 'unauthorized_type'}
        config.limits['models'] = ['authorized_type']
        self.assertRaises(errors.Forbidden, load_model, settings,
                          config=config)

    def test_version_allowed(self):
        self.assertTrue(self.license1.version_allowed())

    def test_version_not_allowed(self):
        loudml.license.LOUDML_MAJOR_VERSION = 2
        self.assertFalse(self.license1.version_allowed())

    def test_host_allowed(self):
        self.assertTrue(self.license1.host_allowed())

    def test_host_now_allowed(self):
        self.license1.payload['hostid'] = '404'
        self.assertFalse(self.license1.host_allowed())

    def test_serial_number(self):
        self.assertEqual(self.license1.payload['serial_num'], "0")

    def test_default_payload(self):
        lic = License()
        payload = lic.default_payload()
        self.assertIn('features', payload)
        self.assertIn('datasources', payload['features'])
        self.assertIn('elasticsearch', payload['features']['datasources'])

    def test_data_range_allowed(self):
        from_date = '2018-01-01'
        to_date = '2018-01-31'
        self.license1.payload['features']['data_range'] = ['2018-01-01', '2018-03-31']
        self.assertTrue(self.license1.data_range_allowed(from_date, to_date))

    def test_data_range_not_allowed(self):
        from_date = '2018-01-01'
        to_date = '2018-01-31'
        self.license1.payload['features']['data_range'] = ['2017-12-31', '2018-01-15']
        self.assertFalse(self.license1.data_range_allowed(from_date, to_date))

import os
import pytest

import loudml

from loudml.license import License

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
@pytest.fixture
def license1():
    l = License()
    l.load(os.path.join(
        os.path.dirname(__file__),
        'resources',
        'license1.lic'
    ))
    return l

def test_expired(license1):
    assert license1.has_expired()

def test_version_allowed(license1):
    assert license1.version_allowed()

def test_version_not_allowed(license1):
    loudml.license.LOUDML_MAJOR_VERSION = 2
    assert not license1.version_allowed()

def test_host_allowed(license1):
    assert license1.host_allowed()

def test_serial_number(license1):
    assert license1.payload['serial_num'] == "0"

def test_default_payload():
    l = License()
    payload = l.default_payload()
    assert 'features' in payload
    assert 'datasources' in payload['features']
    assert 'elasticsearch' in payload['features']['datasources']

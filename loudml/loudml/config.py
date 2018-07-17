"""
LoudML configuration
"""

import copy
import json
import logging
import os
import pkg_resources
import multiprocessing
import voluptuous
import yaml

from . import (
    errors,
)

from loudml.license import License

DEFAULT_MAX_RUNNING_MODELS = 3
DEFAULT_LICENSE_PATH = '/etc/loudml/license.lic'

class Config:
    """
    LoudML configuration
    """

    def __init__(self, data):
        self._data = data

        # TODO check configuration validity with voluptuous

        self._datasources = {
            datasource['name']: datasource
            for datasource in data.get('datasources', [])
        }

        self._license = data.get('license', {})
        if 'path' not in self._license:
            self._license['path'] = None
        if self._license['path'] is None and os.path.isfile(DEFAULT_LICENSE_PATH):
            self._license['path'] = DEFAULT_LICENSE_PATH

        self._storage = data.get('storage', {})
        if 'path' not in self._storage:
            self._storage['path'] = "/var/lib/loudml"

        self._server = data.get('server', {})
        if 'listen' not in self._server:
            self._server['listen'] = "localhost:8077"
        if 'workers' not in self._server:
            self._server['workers'] = multiprocessing.cpu_count()
        if 'maxtasksperchild' not in self._server:
            self._server['maxtasksperchild'] = 1
        self._server['maxrunningmodels'] = DEFAULT_MAX_RUNNING_MODELS
        self._server['allowed_models'] = ['timeseries']


    @property
    def datasources(self):
        # XXX: return a copy to prevent modification by the caller
        return copy.deepcopy(self._datasources)

    @property
    def license(self):
        return self._license

    @property
    def storage(self):
        # XXX: return a copy to prevent modification by the caller
        return copy.deepcopy(self._storage)

    @property
    def server(self):
        # XXX: return a copy to prevent modification by the caller
        return copy.deepcopy(self._server)

    def get_datasource(self, name):
        """
        Get data source configuration by name
        """
        try:
            # XXX: return a copy to prevent modification by the caller
            return copy.deepcopy(self.datasources[name])
        except KeyError:
            raise errors.DataSourceNotFound(name)


    def set_limits(self, license_file):
        """
        Enforce limitations described in license file

        :param license_file: path to license file
        :type  license_file: str

        :raise Exception: when unable to validate license

        If no license file is provided, defaults are used.
        """

        # Keep defaults
        if license_file is None:
            return

        l = License()
        l.load(license_file)
        if not l.validate():
            raise Exception("unable to validate license")

        if l.has_expired():
            logging.warning("license has expired")

        if not l.version_allowed():
            raise Exception("software version not allowed")

        if not l.host_allowed():
            raise Exception("host_id not allowed")

        data = json.loads(l.data.decode('ascii'))
        limits = data['features']
        self.server['maxrunningmodels'] = limits['nrmodels']
        if limits['fingerprints']:
            self.server['allowed_models'].append('fingerprints')


def load_config(config_path):
    """
    Load configuration file
    """
    try:
        with open(config_path) as config_file:
            config_data = yaml.load(config_file)
    except OSError as exn:
        raise errors.LoudMLException(exn)
    except yaml.YAMLError as exn:
        raise errors.LoudMLException(exn)

    config = Config(config_data)

    license_path = config.license['path']
    try:
        config.set_limits(license_path)
    except FileNotFoundError as e:
        raise errors.LoudMLException("Unable to read license file " + license_path + str(e))
    except Exception as e:
        raise errors.LoudMLException("License error " + license_path + ": " + str(e))

    return config


def load_plugins(path):
    """
    Load plug-ins
    """

    if not os.path.isdir(path):
        path = os.path.dirname(path)

    for ep in pkg_resources.iter_entry_points('loudml.plugins'):
        try:
            ep.load()(ep.name, path)
        except OSError as exn:
            raise errors.LoudMLException(exn)
        except yaml.YAMLError as exn:
            raise errors.LoudMLException(exn)
        except voluptuous.Invalid as exn:
            raise errors.Invalid(
                exn.error_message,
                name="{} plug-in configuration".format(ep.name),
                path=exn.path,
            )

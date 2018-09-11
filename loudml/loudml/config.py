"""
LoudML configuration
"""

import copy
import os
import pkg_resources
import multiprocessing
import voluptuous
import yaml

from . import (
    errors,
)

from loudml.license import License

DEFAULT_LICENSE_PATH = '/etc/loudml/license.lic'


class Config:
    """
    LoudML configuration
    """

    def __init__(self, data):
        self._data = data
        self.limits = {}
        self.license = None

        # TODO check configuration validity with voluptuous

        self._datasources = {
            datasource['name']: datasource
            for datasource in data.get('datasources', [])
        }

        self._license = data.get('license', {})
        if 'path' not in self._license:
            self._license['path'] = None
        if (self._license['path'] is None and
                os.path.isfile(DEFAULT_LICENSE_PATH)):
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
        if 'mse_rtol' not in self._server:
            self._server['mse_rtol'] = 4

    @property
    def datasources(self):
        # XXX: return a copy to prevent modification by the caller
        return copy.deepcopy(self._datasources)

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
            datasource = copy.deepcopy(self.datasources[name])
            datasource['allowed'] = copy.deepcopy(self.limits['datasources'])
            return datasource
        except KeyError:
            raise errors.DataSourceNotFound(name)

    def load_license(self):
        """
        Enforce limitations described in license file

        :raise Exception: when unable to validate license

        If no license file is provided, defaults are used.
        """
        path = self._license['path']
        lic = License()

        if path is None:
            self.license_payload = lic.default_payload()
            self.limits = self.license_payload['features']
        else:
            try:
                lic.load(path)
                lic.global_check()
                self.limits = lic.payload['features']
                self.license_payload = lic.payload
            except FileNotFoundError as e:
                raise errors.LoudMLException(
                    "Unable to read license file " + path + str(e))
            except Exception as e:
                raise errors.LoudMLException(
                    "License error " + path + ": " + str(e))

        self.license = lic


def load_config(path):
    """
    Load configuration file
    """
    try:
        with open(path) as config_file:
            config_data = yaml.load(config_file)
    except OSError as exn:
        raise errors.LoudMLException(exn)
    except yaml.YAMLError as exn:
        raise errors.LoudMLException(exn)

    config = Config(config_data)
    config.load_license()

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

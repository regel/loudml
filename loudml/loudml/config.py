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
            return copy.deepcopy(self.datasources[name])
        except KeyError:
            raise errors.DataSourceNotFound(name)


def load_config(path):
    """
    Load configuration file
    """
    try:
        with open(path) as config_file:
            config = yaml.load(config_file)
    except OSError as exn:
        raise errors.LoudMLException(exn)
    except yaml.YAMLError as exn:
        raise errors.LoudMLException(exn)

    return Config(config)

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

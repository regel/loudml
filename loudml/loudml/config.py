"""
Loud ML configuration
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
    Loud ML configuration
    """

    def __init__(self, data):
        self._data = data

        # TODO check configuration validity with voluptuous

        self._datasources = {
            datasource['name']: datasource
            for datasource in data.get('datasources', [])
        }

        self._metrics = data.get('metrics', {})
        if 'enable' not in self._metrics:
            self._metrics['enable'] = True

        self._storage = data.get('storage', {})
        if 'path' not in self._storage:
            self._storage['path'] = "/var/lib/loudml"

        self._training = data.get('training', {})
        if 'num_cpus' not in self._training:
            self._training['num_cpus'] = 1
        if 'num_gpus' not in self._training:
            self._training['num_gpus'] = 0
        if 'nice' not in self._training:
            self._training['nice'] = 5
        if 'batch_size' not in self._training:
            self._training['batch_size'] = 64
        if 'epochs' not in self._training:
            self._training['epochs'] = 100

        if 'incremental' not in self._training:
            self._training['incremental'] = {
                'enable': False,
                'crons': [],
            }

        self._inference = data.get('inference', {})
        if 'num_cpus' not in self._inference:
            self._inference['num_cpus'] = 1
        if 'num_gpus' not in self._inference:
            self._inference['num_gpus'] = 0

        self._server = data.get('server', {})
        if 'listen' not in self._server:
            self._server['listen'] = "localhost:8077"
        if 'workers' not in self._server:
            self._server['workers'] = multiprocessing.cpu_count()
        if 'maxtasksperchild' not in self._server:
            self._server['maxtasksperchild'] = 100

    @property
    def datasources(self):
        # XXX: return a copy to prevent modification by the caller
        return copy.deepcopy(self._datasources)

    @property
    def training(self):
        # XXX: return a copy to prevent modification by the caller
        return copy.deepcopy(self._training)

    @property
    def inference(self):
        # XXX: return a copy to prevent modification by the caller
        return copy.deepcopy(self._inference)

    @property
    def metrics(self):
        return copy.deepcopy(self._metrics)

    @property
    def storage(self):
        # XXX: return a copy to prevent modification by the caller
        return copy.deepcopy(self._storage)

    @property
    def server(self):
        # XXX: return a copy to prevent modification by the caller
        return copy.deepcopy(self._server)

    def put_datasource(self, source):
        """
        Add data source configuration by name
        """
        name = source['name']
        self._datasources[name] = source

    def del_datasource(self, name):
        """
        Del data source configuration by name
        """
        del self._datasources[name]

    def get_datasource(self, name):
        """
        Get data source configuration by name
        """
        try:
            # XXX: return a copy to prevent modification by the caller
            datasource = copy.deepcopy(self.datasources[name])
            return datasource
        except KeyError:
            raise errors.DataSourceNotFound(name)


def load_config(path):
    """
    Load configuration file
    """
    try:
        with open(path) as config_file:
            config_data = yaml.safe_load(config_file)
    except OSError as exn:
        raise errors.LoudMLException(exn)
    except yaml.YAMLError as exn:
        raise errors.LoudMLException(exn)

    config = Config(config_data)

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

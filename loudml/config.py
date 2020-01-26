"""
Loud ML configuration
"""

import copy
import os
import pkg_resources
import multiprocessing
import voluptuous
import yaml
from itertools import chain

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

        self._buckets = {
            bucket['name']: bucket
            for bucket in list(chain(
                data.get('buckets', []),
                data.get('datasources', []),
            ))
        }
        self._scheduled_jobs = {
            scheduled_job['name']: scheduled_job
            for scheduled_job in data.get('scheduled_jobs', [])
        }

        self._cluster = data.get('cluster', {})
        if 'name' not in self._cluster:
            self._cluster['name'] = 'loudml'

        self._node = data.get('node', {})
        if 'name' not in self._node:
            self._node['name'] = 'loudml'
        if 'master' not in self._node:
            self._node['master'] = True
        if 'compute' not in self._node:
            self._node['compute'] = True

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
        if 'jobs_max_ttl' not in self._server:
            self._server['jobs_max_ttl'] = 60

        self._debug = bool(data.get('debug', False))

    @property
    def cluster_name(self):
        return self._cluster['name']

    @property
    def node_name(self):
        return self._node['name']

    @property
    def node(self):
        return copy.deepcopy(self._node)

    def get_node_roles(self):
        roles = []
        for role in ['master', 'compute']:
            if self._node[role]:
                roles.append(role)
        return roles

    @property
    def debug(self):
        return self._debug

    @property
    def scheduled_jobs(self):
        # XXX: return a copy to prevent modification by the caller
        return copy.deepcopy(self._scheduled_jobs)

    @property
    def datasources(self):
        # XXX: return a copy to prevent modification by the caller
        return copy.deepcopy(self._buckets)

    @property
    def buckets(self):
        # XXX: return a copy to prevent modification by the caller
        return copy.deepcopy(self._buckets)

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

    def list_buckets(self):
        return list(self.buckets.keys())

    def put_bucket(self, bucket):
        """
        Add bucket configuration by name
        """
        name = bucket['name']
        self._buckets[name] = bucket

    def del_bucket(self, name):
        """
        Del bucket configuration by name
        """
        try:
            del self._buckets[name]
        except KeyError:
            raise errors.BucketNotFound(name)

    def get_bucket(self, name):
        """
        Get bucket configuration by name
        """
        try:
            # XXX: return a copy to prevent modification by the caller
            bucket = copy.deepcopy(self.buckets[name])
            return bucket
        except KeyError:
            raise errors.BucketNotFound(name)


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

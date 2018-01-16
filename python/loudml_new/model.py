"""
LoudML model
"""

import copy
import pkg_resources

import numpy as np

from . import (
    errors,
)

class Feature:
    """
    Model feature
    """

    def __init__(
        self,
        name,
        metric,
        measurement=None,
        field=None,
        default=None,
        script=None,
    ):
        # TODO use voluptuous to check field validity

        self.name = name
        self.metric = metric
        self.measurement = measurement
        self.field = field
        self.default = np.nan if default is None else default
        self.script = script


class Model:
    """
    LoudML model
    """

    def __init__(self, settings, state=None):
        """
        name -- model settings
        """

        # TODO use voluptuous to check field validity

        settings = copy.deepcopy(settings)
        settings['type'] = 'timeseries'
        self._settings = settings
        self.name = settings.get('name')
        self._settings = settings
        self.index = settings.get('index')
        self.db = settings.get('db')
        self.measurement = settings.get('measurement')
        self.routing = settings.get('routing')
        self.state = state
        self.features = [Feature(**feature) for feature in settings['features']]

    @property
    def type(self):
        return self.settings['type']

    @property
    def settings(self):
        return self._settings

    @property
    def data(self):
        return {
            'settings': self.settings,
            'state': self.state,
        }

def load_model(settings, state=None):
    """
    Load model
    """

    model_type = settings['type']
    for ep in pkg_resources.iter_entry_points('loudml.models', model_type):
        if ep.name == model_type:
            return ep.load()(settings, state)
    raise errors.UnsupportedModel(model_type)

"""
LoudML model
"""

import copy
import pkg_resources

from . import (
    errors,
)

class Model:
    """
    LoudML model
    """

    def __init__(self, settings, state=None):
        """
        name -- model settings
        """

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

    @property
    def type(self):
        return self.settings['type']

    @property
    def features(self):
        """Model features"""
        return self.settings['features']

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

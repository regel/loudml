"""
LoudML model
"""

import copy
import pkg_resources

import numpy as np

from voluptuous import (
    ALLOW_EXTRA,
    All,
    Any,
    Length,
    Required,
    Schema,
)

from . import (
    errors,
    schemas,
)

class Feature:
    """
    Model feature
    """

    SCHEMA = Schema({
        Required('name'): schemas.key,
        Required('metric'): schemas.key,
        Required('field'): schemas.key,
        'measurement': Any(None, schemas.key),
        'default': Any(None, int, float),
        'script': Any(None, str),
    })

    def __init__(
        self,
        name=None,
        metric=None,
        field=None,
        measurement=None,
        default=None,
        script=None,
    ):
        self.validate(locals())

        self.name = name
        self.metric = metric
        self.measurement = measurement
        self.field = field
        self.default = np.nan if default is None else float(default)
        self.script = script

    @classmethod
    def validate(cls, args):
        del args['self']
        schemas.validate(cls.SCHEMA, args)


class Model:
    """
    LoudML model
    """

    SCHEMA = Schema({
        Required('name'): schemas.key,
        Required('type'): schemas.key,
        Required('features'): All([Feature.SCHEMA], Length(min=1)),
        'routing': Any(None, schemas.key),
    }, extra=ALLOW_EXTRA)

    def __init__(self, settings, state=None):
        """
        name -- model settings
        """

        self.validate(settings)
        settings = copy.deepcopy(settings)

        self._settings = settings
        self.name = settings.get('name')
        self._settings = settings
        self.routing = settings.get('routing')
        self.state = state
        self.features = [Feature(**feature) for feature in settings['features']]

    @classmethod
    def validate(cls, settings):
        """Validate the settings against the schema"""
        schemas.validate(cls.SCHEMA, settings)

    @property
    def type(self):
        return self.settings['type']

    @property
    def default_datasource(self):
        return self._settings.get('default_datasource')

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

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
    Range,
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
        Required('name'): All(schemas.key, Length(max=256)),
        Required('metric'): All(schemas.key, Length(max=256)),
        Required('field'): All(schemas.key, Length(max=256)),
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

    TYPE = 'generic'
    SCHEMA = Schema({
        Required('name'): All(schemas.key, Length(max=256)),
        Required('type'): All(schemas.key, Length(max=256)),
        Required('features'): All([Feature.SCHEMA], Length(min=1)),
        'routing': Any(None, schemas.key),
        'threshold': Any(int, float, Range(min=0, max=100)),
    }, extra=ALLOW_EXTRA)

    def __init__(self, settings, state=None):
        """
        name -- model settings
        """

        settings['type'] = self.TYPE
        settings = copy.deepcopy(settings)

        self._settings = self.validate(settings)
        self.name = settings.get('name')
        self._settings = settings
        self.routing = settings.get('routing')
        self.state = state
        self.features = [Feature(**feature) for feature in settings['features']]
        self.threshold = self.settings.get('threshold', 75)

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
    def is_trained(self):
        return self.state is not None

    @property
    def data(self):
        return {
            'settings': self.settings,
            'state': self.state,
        }

    @property
    def preview(self):
        state = {
            'trained': self.is_trained,
        }

        if self.is_trained:
            state['loss'] = self.state.get('loss')
            state['accuracy'] = self.state.get('accuracy')

        return {
            'settings': self.settings,
            'state': state,
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

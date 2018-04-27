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
    Optional,
    Boolean,
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
        'match_all': Any(None, Schema([
            {Required(schemas.key): Any(
                Boolean(),
                int,
                float,
                All(str, Length(max=256)),
            )},
        ])),
        'default': Any(None, int, float),
        'script': Any(None, str),
    })

    def __init__(
        self,
        name=None,
        metric=None,
        field=None,
        measurement=None,
        match_all=None,
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
        self.match_all = match_all

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
        Optional('features'): All([Feature.SCHEMA], Length(min=1)),
        Optional('influences'): Any(None, All([Feature.SCHEMA], Length(min=1))),
        'routing': Any(None, schemas.key),
        'threshold': Any(int, float, Range(min=0, max=100)),
        'max_evals': All(int, Range(min=1)),
    }, extra=ALLOW_EXTRA)

    def __init__(self, settings, state=None):
        """
        name -- model settings
        """

        settings['type'] = self.TYPE
        settings = copy.deepcopy(settings)

        settings = self.validate(settings)
        self._settings = settings
        self.name = settings.get('name')
        self._settings = settings
        self.routing = settings.get('routing')
        self._state = state

        features = settings.get('features')
        if features is None:
            self.features = None
        else:
            self.features = [Feature(**feature) for feature in features]

        influences = settings.get('influences')
        if influences is None:
            self.influences = []
        else:
            self.influences = [Feature(**feature) for feature in influences]

        self.threshold = self.settings.get('threshold', 75)

    @classmethod
    def validate(cls, settings):
        """Validate the settings against the schema"""
        return schemas.validate(cls.SCHEMA, settings)

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
    def nb_influences(self):
        return len(self.influences)

    @property
    def nb_features(self):
        return len(self.features)

    @property
    def is_trained(self):
        return self._state is not None

    @property
    def data(self):
        return {
            'settings': self.settings,
            'state': self.state,
        }

    @property
    def seasonality(self):
        return self._settings['seasonality']

    @property
    def state(self):
        return self._state

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

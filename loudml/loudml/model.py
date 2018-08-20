"""
LoudML model
"""

import copy
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
    misc,
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
                int,
                bool,
                float,
                All(str, Length(max=256)),
            )},
        ])),
        'default': Any(None, int, float, 'previous'),
        'script': Any(None, str),
        Optional('anomaly_type', default='low_high'): Any('low', 'high', 'low_high'),
        'transform': Any(None, "diff"),
        'scores': Any(None, "min_max", "normalize", "standardize"),
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
        anomaly_type='low_high',
        transform=None,
        scores=None,
    ):
        self.validate(locals())

        self.name = name
        self.metric = metric
        self.measurement = measurement
        self.field = field
        self.default = np.nan if default is None else default
        self.script = script
        self.match_all = match_all
        self.anomaly_type = anomaly_type
        self.is_input = True
        self.is_output = True
        self.transform = transform
        self.scores = "min_max" if scores is None else scores

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
        Optional('features'): Any(None,
            All([Feature.SCHEMA], Length(min=1)),
            Schema({
                Optional('i'): All([Feature.SCHEMA], Length(min=1)),
                Optional('o'): All([Feature.SCHEMA], Length(min=1)),
                Optional('io'): All([Feature.SCHEMA], Length(min=1)),
            }),
        ),
        'routing': Any(None, schemas.key),
        'threshold': schemas.score,
        'max_threshold': schemas.score,
        'min_threshold': schemas.score,
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
        if isinstance(features, list):
            self.features = [Feature(**feature) for feature in features]
        elif isinstance(features, dict):
            _in = features.get('i')
            _out = features.get('o')
            _in_out = features.get('io')
            if _in is None:
                _in = []
            else:
                _in = [Feature(**feature) for feature in _in]
                for feature in _in:
                    feature.is_input = True
                    feature.is_output = False

            if _out is None:
                _out = []
            else:
                _out = [Feature(**feature) for feature in _out]
                for feature in _out:
                    feature.is_input = False
                    feature.is_output = True

            if _in_out is None:
                _in_out = []
            else:
                _in_out = [Feature(**feature) for feature in _in_out]

            self.features = _in_out + _out + _in

        self.max_threshold = self.settings.get('max_threshold')
        if self.max_threshold is None:
            # Backward compatibility
            self.max_threshold = self.settings.get('threshold', 0)
            self.settings['max_threshold'] = self.max_threshold

        self.min_threshold = self.settings.get('min_threshold')
        if self.min_threshold is None:
            # Backward compatibility
            self.min_threshold = self.settings.get('threshold', 0)
            self.settings['min_threshold'] = self.min_threshold

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

        return {
            'settings': self.settings,
            'state': state,
        }

    def generate_fake_prediction(self):
        """
        Generate a prediction with fake values. Just for testing purposes.
        """
        return NotImplemented()

def load_model(settings, state=None, config=None):
    """
    Load model

    :param settings: model settings
    :type  settings: dict

    :param state: model state
    :type  state: opaque type

    :param config: running configuration
    :type  config: loudml.Config
    """

    model_type = settings['type']

    if config and model_type not in config.limits['models']:
        raise errors.Forbidden("Not allowed by license: " + model_type)

    try:
        model_cls = misc.load_entry_point('loudml.models', model_type)
    except ImportError:
        raise errors.UnsupportedModel(model_type)

    if model_cls is None:
        raise errors.UnsupportedModel(model_type)
    return model_cls(settings, state)

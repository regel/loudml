"""
Loud ML model
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
    Schema,
)

import loudml
from loudml import (
    errors,
    misc,
    schemas,
)

import json
from jinja2 import Template


def _convert_features_dict(features):
    """
    Convert old features dict format to list
    """

    result = []

    for io, lst in features.items():
        for feature in lst:
            feature['io'] = io
            result.append(feature)

    return result


def flatten_features(features):
    """
    Normalize feature list to the current format
    """

    if isinstance(features, dict):
        features = _convert_features_dict(features)

    inout = []
    in_only = []
    out_only = []

    for feature in features:
        io = feature.get('io')

        if io == 'o':
            out_only.append(feature)
        elif io == 'i':
            in_only.append(feature)
        else:
            if io is None:
                feature['io'] = 'io'
            inout.append(feature)

    return inout + out_only + in_only


class Feature:
    """
    Model feature
    """

    SCHEMA = Schema({
        Required('name'): All(schemas.key, Length(max=256)),
        Required('metric'): All(schemas.key, Length(max=256)),
        Required('field'): All(schemas.dotted_key, Length(max=256)),
        'bucket': Any(None, schemas.key),
        'measurement': Any(None, schemas.dotted_key),
        'match_all': Any(None, Schema([
            {Required(schemas.key): Any(
                int,
                bool,
                float,
                All(str, Length(max=256)),
            )},
        ])),
        'default': Any(None, int, float, 'previous'),
        Optional('io', default='io'): Any('io', 'o', 'i'),
        'script': Any(None, str),
        Optional('anomaly_type', default='low_high'):
            Any('low', 'high', 'low_high'),
        'transform': Any(None, "diff"),
        'scores': Any(None, "min_max", "normalize", "standardize"),
    }, extra=ALLOW_EXTRA)

    def __init__(
        self,
        name=None,
        metric=None,
        field=None,
        bucket=None,
        measurement=None,
        match_all=None,
        default=None,
        script=None,
        anomaly_type='low_high',
        transform=None,
        scores=None,
        io='io',
    ):
        self.validate(locals())

        self.name = name
        self.metric = metric
        self.bucket = bucket
        self.measurement = measurement
        self.field = field
        self.default = np.nan if default is None else default
        self.script = script
        self.match_all = match_all
        self.anomaly_type = anomaly_type
        self.is_input = 'i' in io
        self.is_output = 'o' in io
        self.transform = transform
        self.scores = "min_max" if scores is None else scores
        self.agg_id = self.build_agg_id()

    def build_agg_id(self):
        prefix = self.measurement

        if not self.match_all:
            return prefix or 'all'

        return "{}_{}".format(
            prefix,
            misc.hash_dict(self.match_all)
        )

    @classmethod
    def validate(cls, args):
        del args['self']
        return schemas.validate(cls.SCHEMA, args)


class FeatureTemplate(Feature):
    SCHEMA = Schema({
        Required('name'): Any(
            All(schemas.key, Length(max=256)),
            All(schemas.bracket_key, Length(max=256)),
        ),
        Required('metric'): Any(
            All(schemas.key, Length(max=256)),
            All(schemas.bracket_key, Length(max=256)),
        ),
        Required('field'): Any(
            All(schemas.dotted_key, Length(max=256)),
            All(schemas.bracket_key, Length(max=256)),
        ),
        'bucket': Any(
            None, schemas.key, schemas.bracket_key),
        'measurement': Any(
            None, schemas.dotted_key, schemas.bracket_key),
        'match_all': Any(
            None,
            Schema([
                {Required(schemas.key): Any(
                    int,
                    bool,
                    float,
                    All(str, Length(max=256)),
                )}]),
            Schema([
                {Required(schemas.bracket_key): Any(
                    int,
                    bool,
                    float,
                    All(str, Length(max=256)),
                )}]),
        ),
        'default': Any(
            None, int, float, 'previous', schemas.bracket_key),
        Optional('io', default='io'): Any(
            'io', 'o', 'i', schemas.bracket_key),
        'script': Any(None, str, schemas.bracket_key),
        Optional('anomaly_type', default='low_high'):
            Any('low', 'high', 'low_high', schemas.bracket_key),
        'transform': Any(None, "diff", schemas.bracket_key),
        'scores': Any(
            None,
            "min_max",
            "normalize",
            "standardize",
            schemas.bracket_key,
        ),
    })

    @classmethod
    def validate(cls, args):
        del args['self']
        return schemas.validate(cls.SCHEMA, args)


class Model:
    """
    Loud ML model
    """

    TYPE = 'model_cls'
    SCHEMA = Schema({
        Required('name'): All(schemas.key, Length(max=256)),
        Required('type'): All(schemas.key, Length(max=256)),
        Optional('features'):
            Any(None,
                All([Feature.SCHEMA], Length(min=1)),
                Schema({
                    Optional('i'): All([Feature.SCHEMA], Length(min=1)),
                    Optional('o'): All([Feature.SCHEMA], Length(min=1)),
                    Optional('io'): All([Feature.SCHEMA], Length(min=1)),
                }),
                ),
        Optional('bucket_interval'): schemas.TimeDelta(
            min=0, min_included=False,
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
        self.routing = settings.get('routing')
        self._state = state

        self.features = [
            Feature(**feature) for feature in settings['features']
        ]
        self.bucket_interval = misc.parse_timedelta(
            settings.get('bucket_interval', 0)).total_seconds()

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

        res = schemas.validate(cls.SCHEMA, settings)

        features = flatten_features(settings.get('features'))
        res['features'] = features

        has_input = False
        has_output = False

        for feature in res['features']:
            io = feature.get('io', 'io')
            if 'i' in io:
                has_input = True
            if 'o' in io:
                has_output = True
            if has_input and has_output:
                break

        if not has_input:
            raise errors.Invalid('model has no input feature')
        if not has_output:
            raise errors.Invalid('model has no output feature')

        return res

    @property
    def type(self):
        return self.settings['type']

    @property
    def default_bucket(self):
        return self._settings.get('default_bucket')

    def get_tags(self):
        tags = {
            'model': self.name,
        }
        for feature in self.features:
            if feature.match_all:
                for condition in feature.match_all:
                    tag = condition['tag']
                    val = condition['value']
                    tags[tag] = val

        return tags

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


class ModelTemplate(Model):
    """
    Loud ML Jinja model template
    """

    TYPE = 'template_cls'
    SCHEMA = Schema({
        Required('name'): Any(
            All(schemas.key, Length(max=256)),
            All(schemas.bracket_key, Length(max=256)),
        ),
        Required('type'): Any(
            All(schemas.key, Length(max=256)),
            All(schemas.bracket_key, Length(max=256)),
        ),
        Optional('features'):
            Any(None,
                All([FeatureTemplate.SCHEMA], Length(min=1)),
                ),
        Optional('bucket_interval'): Any(
            schemas.TimeDelta(min=0, min_included=False),
            All(schemas.bracket_key),
        ),
        'threshold': Any(
            schemas.score,
            All(schemas.bracket_key),
        ),
        'max_threshold': Any(
            schemas.score,
            All(schemas.bracket_key),
        ),
        'min_threshold': Any(
            schemas.score,
            All(schemas.bracket_key),
        ),
        'max_evals': Any(
            All(int, Range(min=1)),
            All(schemas.bracket_key),
        ),
    }, extra=ALLOW_EXTRA)

    def __init__(self, settings, name):
        settings = copy.deepcopy(settings)

        settings = self.validate(settings)
        self._settings = settings
        self.name = name
        self._state = None

        self.features = [
            FeatureTemplate(**feature)
            for feature in settings['features']
        ]
        self.bucket_interval = misc.parse_timedelta(
            settings.get('bucket_interval', 0)).total_seconds()

    @classmethod
    def validate(cls, settings):
        """Validate the settings against the schema"""
        return schemas.validate(cls.SCHEMA, settings)

    @property
    def is_trained(self):
        return False

    @property
    def data(self):
        return {
            'settings': self.settings,
        }

    @property
    def state(self):
        return None

    @property
    def preview(self):
        return {
            'settings': self.settings,
        }


def load_model(settings, state=None):
    """
    Load model

    :param settings: model settings
    :type  settings: dict

    :param state: model state
    :type  state: opaque type
    """

    model_type = settings.get('type')

    if model_type is None:
        raise errors.Invalid("model has no type")

    try:
        model_cls = loudml.load_entry_point('loudml.models', model_type)
    except ImportError:
        raise errors.UnsupportedModel(model_type)

    if model_cls is None:
        raise errors.UnsupportedModel(model_type)
    return model_cls(settings, state)


def load_template(settings, name):
    return ModelTemplate(settings, name)


def load_model_from_template(settings, state=None, *args, **kwargs):
    t = Template(json.dumps(settings))
    settings = json.loads(t.render(**kwargs))
    return load_model(settings, state)

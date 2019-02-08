"""
Loud ML model
"""

import copy
import math
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

from . import (
    errors,
    misc,
    schemas,
)

import json
from jinja2 import Template
from jinja2 import Environment, meta


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


class DateRange:
    def __init__(self, from_date, to_date):
        self.from_ts = misc.make_ts(from_date)
        self.to_ts = misc.make_ts(to_date)

        if self.to_ts < self.from_ts:
            raise errors.Invalid("invalid date range: {}".format(self))

    @property
    def from_str(self):
        return misc.ts_to_str(self.from_ts)

    @property
    def to_str(self):
        return misc.ts_to_str(self.to_ts)

    def __str__(self):
        return "{}-{}".format(
            self.from_str,
            self.to_str,
        )


class Feature:
    """
    Model feature
    """

    SCHEMA = Schema({
        Required('name'): All(schemas.key, Length(max=256)),
        Required('metric'): All(schemas.key, Length(max=256)),
        Required('field'): All(schemas.dotted_key, Length(max=256)),
        'measurement': Any(None, schemas.dotted_key),
        'collection': Any(None, schemas.key),
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
        'low_watermark': Any(None, int, float),
        'high_watermark': Any(None, int, float),
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
        collection=None,
        match_all=None,
        default=None,
        script=None,
        anomaly_type='low_high',
        transform=None,
        scores=None,
        io='io',
        low_watermark=None,
        high_watermark=None,
    ):
        self.validate(locals())

        self.name = name
        self.metric = metric
        self.measurement = measurement
        self.collection = collection
        self.field = field
        self.default = np.nan if default is None else default
        self.low_watermark = low_watermark
        self.high_watermark = high_watermark
        self.script = script
        self.match_all = match_all
        self.anomaly_type = anomaly_type
        self.is_input = 'i' in io
        self.is_output = 'o' in io
        self.transform = transform
        self.scores = "min_max" if scores is None else scores
        self.agg_id = self.build_agg_id()

    def build_agg_id(self):
        prefix = self.measurement or self.collection

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


class Model:
    """
    Loud ML model
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
        Optional('bucket_interval'): schemas.TimeDelta(
            min=0, min_included=False,
        ),
        'timestamp_field': schemas.key,
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
        self.timestamp_field = settings.get('timestamp_field', 'timestamp')
        self.bucket_interval = misc.parse_timedelta(settings.get('bucket_interval', 0)).total_seconds()

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

    def build_date_range(self, from_date, to_date):
        """
        Fixup date range to be sure that is a multiple of bucket_interval

        return timestamps
        """

        from_ts = misc.make_ts(from_date)
        to_ts = misc.make_ts(to_date)

        from_ts = math.floor(from_ts / self.bucket_interval) * self.bucket_interval
        to_ts = math.ceil(to_ts / self.bucket_interval) * self.bucket_interval

        return DateRange(from_ts, to_ts)

    @property
    def type(self):
        return self.settings['type']

    @property
    def default_datasource(self):
        return self._settings.get('default_datasource')

    @property
    def default_datasink(self):
        return self._settings.get('default_datasink')

    def get_tags(self):
        tags = {}
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

    model_type = settings.get('type')

    if model_type is None:
        raise errors.Invalid("model has no type")

    try:
        model_cls = misc.load_entry_point('loudml.models', model_type)
    except ImportError:
        raise errors.UnsupportedModel(model_type)

    if model_cls is None:
        raise errors.UnsupportedModel(model_type)
    return model_cls(settings, state)


def load_template(settings, state=None, config=None, *args, **kwargs):
    t = Template(json.dumps(settings))
    settings = json.loads(t.render(**kwargs))
    return load_model(settings, state, config)


def find_undeclared_variables(settings):
    env = Environment()
    ast = env.parse(json.dumps(settings))
    return meta.find_undeclared_variables(ast)

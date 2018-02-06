"""
LoudML fingerprints module
"""

import copy
import json
import logging
import math

import numpy as np

from voluptuous import (
    All,
    Length,
    Range,
    Required,
)

from . import (
    errors,
    schemas,
    som,
)
from .misc import (
    make_ts,
    parse_timedelta,
    ts_to_str,
)
from .model import Model

class FingerprintsPrediction:
    def __init__(self, fingerprints, from_ts, to_ts):
        self.from_ts = from_ts
        self.to_ts = to_ts
        self.fingerprints = fingerprints

    def format(self):
        return {
            'from_date': ts_to_str(self.from_ts),
            'to_date': ts_to_str(self.to_ts),
            'fingerprints': self.fingerprints,
        }

    def __str__(self):
        return json.dumps(self.format(), indent=4)


class FingerprintsModel(Model):
    """
    Fingerprints model
    """

    TYPE = 'fingerprints'

    SCHEMA = Model.SCHEMA.extend({
        Required('key'): All(schemas.key, Length(max=256)),
        Required('max_keys'): All(int, Range(min=1)),
        Required('width'): All(int, Range(min=1)),
        Required('height'): All(int, Range(min=1)),
        Required('interval'): schemas.TimeDelta(min=0, min_included=False),
        'timestamp_field': schemas.key,
    })

    def __init__(self, settings, state=None):
        super().__init__(settings, state)

        self.key = settings['key']
        self.max_keys = settings['max_keys']
        self.w = settings['width']
        self.h = settings['height']
        self.interval = parse_timedelta(settings['interval']).total_seconds()
        self.timestamp_field = settings.get('timestamp_field', 'timestamp')

        if state is not None:
            self._state = state
            self._means = np.array(state['means'])
            self._stds = np.array(state['stds'])

        self._som_model = None

    @property
    def state(self):
        if self._state is None:
            return None

        # XXX As we add 'means' and 'stds', we need a copy to avoid
        # modifying self._state in-place
        state = copy.deepcopy(self._state)
        state['means'] = self._means.tolist()
        state['stds'] = self._stds.tolist()
        return state

    @property
    def is_trained(self):
        return self._state is not None and 'ckpt' in self._state

    def _format_quadrants(self, time_buckets):
        # TODO generic implementation not yet available
        raise NotImplemented()

    def _train_on_dataset(
        self,
        dataset,
        num_epochs=100,
        limit=-1,
    ):
        # Apply data standardization to each feature individually
        # https://en.wikipedia.org/wiki/Feature_scaling
        self._means = np.mean(dataset, axis=0)
        self._stds = np.std(dataset, axis=0)
        zY = np.nan_to_num((dataset - self._means) / self._stds)

        # Hyperparameters
        data_dimens = self.nb_features
        self._som_model = som.SOM(self.w, self.h, data_dimens, num_epochs)

        # Start Training
        self._som_model.train(zY, truncate=limit)

        # Map vectors to their closest neurons
        return self._som_model.map_vects(zY)

    def _build_fingerprints(
        self,
        dataset,
        mapped,
        keys,
        from_ts,
        to_ts,
    ):
        fingerprints = []

        for i, x in enumerate(mapped):
            key = keys[i]
            _fingerprint = np.nan_to_num((dataset[i] - self._means) / self._stds)
            fingerprints.append({
                'key': key,
                'time_range': (int(from_ts), int(to_ts)),
                'fingerprint': dataset[i].tolist(),
                '_fingerprint': _fingerprint.tolist(),
                'location': (mapped[i][0].item(), mapped[i][1].item()),
            })

        return fingerprints

    def train(
        self,
        datasource,
        from_date=None,
        to_date=None,
        num_epochs=100,
        limit=-1,
    ):
        self._som_model = None
        self._means = None
        self._stds = None

        if from_date:
            from_ts = make_ts(from_date)
        else:
            from_ts = datasource.get_times_start(self.index)

        if to_date:
            to_ts = make_ts(to_date)
        else:
            to_ts = datasource.get_times_end(self.index)

        from_str = ts_to_str(from_ts)
        to_str = ts_to_str(to_ts)

        logging.info(
            "train(%s) range=[%s, %s] epochs=%d limit=%d)",
            self.name,
            from_str,
            to_str,
            num_epochs,
            limit,
        )

        # Prepare dataset
        nb_keys = self.max_keys
        nb_features = self.nb_features
        dataset = np.zeros((nb_keys, nb_features), dtype=float)

        # Fill dataset
        data = datasource.get_quadrant_data(self, from_ts, to_ts)

        i = None
        keys = []
        for i, (key, val) in enumerate(data):
            keys.append(key)
            dataset[i] = self.format_quadrants(val)

        if i is None:
            raise errors.NoData("no data found for time range {}-{}".format(
                from_str,
                to_str,
            ))

        logging.info("found %d keys", i + 1)

        mapped = self._train_on_dataset(
            np.resize(dataset, (i + 1, nb_features)),
            num_epochs,
            limit,
        )

        model_ckpt, model_index, model_meta = som.serialize_model(self._som_model)
        fingerprints = self._build_fingerprints(
            dataset,
            mapped,
            keys,
            from_ts,
            to_ts,
        )

        self._state = {
            'ckpt': model_ckpt, # TF CKPT data encoded in base64
            'index': model_index,
            'meta': model_meta,
            'fingerprints': fingerprints,
        }

    def load(self):
        if not self.is_trained:
            return errors.ModelNotTrained()

        self._som_model = som.load_model(
            self._state['ckpt'],
            self._state['index'],
            self._state['meta'],
            self.w,
            self.h,
            self.nb_features,
        )

    @property
    def preview(self):
        trained = self.is_trained

        state = {
            'trained': self.is_trained
        }

        last_prediction = self._state.get('last_prediction')
        if last_prediction is not None:
            state['last_prediction'] = {
                'from_date': last_prediction['from_date'],
                'to_date': last_prediction['to_date'],
                'fingerprints': len(last_prediction['fingerprints']),
            }

        return {
            'settings': self.settings,
            'state': state,
        }

    def _map_dataset(self, dataset):
        zY = np.nan_to_num((dataset - self._means) / self._stds)

        # Hyperparameters
        mapped = self._som_model.map_vects(zY)
        return mapped

    def predict(
        self,
        datasource,
        from_date,
        to_date,
    ):
        from_ts = make_ts(from_date)
        to_ts = make_ts(to_date)

        # Fixup range to be sure that it is a multiple of interval
        from_ts = math.floor(from_ts / self.interval) * self.interval
        to_ts = math.ceil(to_ts / self.interval) * self.interval

        from_str = ts_to_str(from_ts)
        to_str = ts_to_str(to_ts)

        logging.info("predict(%s) range=[%s, %s]",
                     self.name, from_str, to_str)

        self.load()

        # Prepare dataset
        nb_keys = self.max_keys
        dataset = np.zeros((nb_keys, self.nb_features), dtype=float)

        # Fill dataset
        data = datasource.get_quadrant_data(self, from_ts, to_ts)

        i = None
        keys = []
        for i, (key, val) in enumerate(data):
            keys.append(key)
            dataset[i] = self.format_quadrants(val)

        if i is None:
            raise errors.NoData("no data found for time range {}-{}".format(
                from_str,
                to_str,
            ))

        logging.info("found %d keys", i + 1)

        mapped = self._map_dataset(
            np.resize(dataset, (i + 1, self.nb_features)),
        )

        fingerprints = self._build_fingerprints(
            dataset,
            mapped,
            keys,
            from_ts,
            to_ts,
        )

        return FingerprintsPrediction(
            from_ts=from_ts,
            to_ts=to_ts,
            fingerprints=fingerprints,
        )

    def keep_prediction(self, prediction):
        if not self.is_trained:
            return errors.ModelNotTrained()

        self._state['last_prediction'] = prediction.format()

"""
LoudML fingerprints module
"""

import copy
import logging

import numpy as np

from voluptuous import (
    All,
    Length,
    Range,
    Required,
)

from . import (
    schemas,
    som,
)
from .misc import (
    make_ts,
    ts_to_str,
)
from .model import Model

class FingerprintsModel(Model):
    """
    Fingerprints model
    """

    TYPE = 'fingerprints'

    SCHEMA = Model.SCHEMA.extend({
        Required('term'): All(schemas.key, Length(max=256)),
        Required('max_terms'): All(int, Range(min=1)),
        Required('width'): All(int, Range(min=1)),
        Required('height'): All(int, Range(min=1)),
        'timestamp_field': schemas.key,
    })

    def __init__(self, settings, state=None):
        super().__init__(settings, state)

        self.term = settings['term']
        self.max_terms = settings['max_terms']
        self.w = settings['width']
        self.h = settings['height']
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
        nb_terms = self.max_terms
        nb_features = self.nb_features
        dataset = np.zeros((nb_terms, nb_features), dtype=float)

        # Fill dataset
        data = datasource.get_quadrant_data(self, from_ts, to_ts)

        i = None
        terms = []
        for i, (term_val, val) in enumerate(data):
            terms.append(term_val)
            dataset[i] = self.format_quadrants(val)

        if i is None:
            raise errors.NoData("no data found for time range {}-{}".format(
                from_str,
                to_str,
            ))

        logging.info("found %d terms", i + 1)

        mapped = self._train_on_dataset(
            np.resize(dataset, (i + 1, nb_features)),
            num_epochs,
            limit,
        )

        model_ckpt, model_index, model_meta = som.serialize_model(self._som_model)
        fingerprints = []
        i = None
        for i, x in enumerate(mapped):
            key = terms[i]
            _fingerprint = np.nan_to_num((dataset[i] - self._means) / self._stds)
            fingerprints.append({
                'key': key,
                 'time_range': (int(from_ts), int(to_ts)),
                 'fingerprint': dataset[i].tolist(),
                 '_fingerprint': _fingerprint.tolist(),
                 'location': (mapped[i][0].item(), mapped[i][1].item()),
               })

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

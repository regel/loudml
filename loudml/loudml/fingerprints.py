"""
Loud ML fingerprints module
"""

import loudml.vendor

import operator
import datetime
import copy
import json
import logging
import math
import sys
import os
import copy

from itertools import repeat

assert sys.version.startswith('3')

_MAX_INT = sys.maxsize

import numpy as np

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

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
    som,
)

from .misc import (
    datetime_to_str,
    make_ts,
    parse_timedelta,
    ts_to_str,
    ts_to_datetime,
    build_agg_name,
    Pool,
    chunks,
)
from .model import (
    Model,
    Feature,
)

class Aggregation:
    """
    Aggregation of features matching the same criteria
    """

    def __init__(
        self,
        agg_id,
        measurement=None,
        match_all=None,
    ):
        self.agg_id = agg_id
        self.measurement = measurement
        self.match_all = match_all
        self.features = []


class FingerprintsPrediction:
    def __init__(self, fingerprints, from_ts, to_ts):
        self.from_ts = from_ts
        self.to_ts = to_ts
        self.fingerprints = fingerprints
        self.changed = None
        self.anomalies = None

    def format(self):
        fps = {
            fingerprint['key']: fingerprint
            for fingerprint in self.fingerprints
        }

        result = {
            'from_date': ts_to_str(self.from_ts),
            'to_date': ts_to_str(self.to_ts),
            'fingerprints': fps,
        }
        if self.changed is not None:
            result['changed'] = self.changed
        if self.anomalies is not None:
            result['anomalies'] = self.anomalies
        return result

    def __str__(self):
        return json.dumps(self.format(), indent=4)


def predict_scores(args):
    model, source, key, date_range = args
    model.load()
    try:
        prediction = model.predict(
            source,
            date_range[0],
            date_range[1],
            key,
        )
    except errors.NoData as exn:
        logging.warning(exn)
        prediction = FingerprintsPrediction(
                        from_ts=make_ts(date_range[0]),
                        to_ts=make_ts(date_range[1]),
                        fingerprints=[],
                        )

    model.detect_anomalies(prediction)
    model.unload()
    return prediction


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
        Required('span'): schemas.TimeDelta(min=0, min_included=False),
        Optional('bucket_interval', default=24 * 3600): schemas.TimeDelta(
            min=0,
            min_included=False,
        ),
        'offset': schemas.TimeDelta(min=0),
        'timestamp_field': schemas.key,
    })

    def __init__(self, settings, state=None):
        super().__init__(settings, state)

        settings = self.settings
        self.key = settings['key']
        self.max_keys = settings['max_keys']
        self.w = settings['width']
        self.h = settings['height']
        self.interval = parse_timedelta(settings['interval']).total_seconds()
        bucket_interval = settings.get('bucket_interval')
        self.bucket_interval = parse_timedelta(bucket_interval).total_seconds()
        self.span = parse_timedelta(settings['span']).total_seconds()
        self.offset = parse_timedelta(settings.get('offset', 0)).total_seconds()
        self.timestamp_field = settings.get('timestamp_field', 'timestamp')

        self._aggs = {}
        self.aggs = None
        self._register_aggregations()

        if state is not None:
            self._state = state
            self._means = np.array(state['means'])
            self._stds = np.array(state['stds'])

        self._som_model = None

    def _register_aggregations(self):
        for feature in self.features:
             agg = self._aggs.get(feature.agg_id)
             if agg is None:
                 agg = self._aggs[feature.agg_id] = Aggregation(
                     feature.agg_id,
                     feature.measurement or feature.collection,
                     feature.match_all,
                 )
             agg.features.append(feature)

        self.aggs = [
            self._aggs[agg_id]
            for agg_id in sorted(self._aggs.keys())
        ]

    @property
    def type(self):
        return self.TYPE

    @property
    def nb_quadrants(self):
        return int(24*3600 / self.bucket_interval)

    @property
    def nb_dimensions(self):
        return self.nb_quadrants * self.nb_features

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

    @property
    def feature_names(self):
        return [feature.name for feature in self.features]

    def format_quadrants(self, time_buckets, agg):
        # init: all zeros except the mins
        res = np.zeros(self.nb_quadrants * len(agg.features))
        counts = np.zeros(self.nb_quadrants * len(agg.features))
        sums = np.zeros(self.nb_quadrants * len(agg.features))
        sum_of_squares = np.zeros(self.nb_quadrants * len(agg.features))

        for quad_num in range(self.nb_quadrants):
            for feat_num, feature in enumerate(agg.features):
                quad_pos = quad_num * len(agg.features)
                _pos = quad_pos + feat_num
                if feature.metric == 'min':
                    res[_pos] = _MAX_INT

        for l in time_buckets:
            ts = make_ts(l['key_as_string'])
            quad_num = int((int(ts) / self.bucket_interval)) % self.nb_quadrants
            quad_pos = quad_num * len(agg.features)

            for feat_num, feature in enumerate(agg.features):
                _pos = quad_pos + feat_num
                s = l[build_agg_name(agg.measurement, feature.field)]
                _count = float(s['count'])
                if _count != 0:
                    _min = float(s['min'])
                    _max = float(s['max'])
                    _avg = float(s['avg'])
                    _sum = float(s['sum'])
                    _sum_of_squares = float(s['sum_of_squares'])
                    _variance = float(s['variance'])
                    _std_deviation = float(s['std_deviation'])

                    counts[_pos] = counts[_pos] + _count
                    sums[_pos] = sums[_pos] + _sum
                    sum_of_squares[_pos] = sum_of_squares[_pos] + _sum_of_squares
                    if feature.metric == 'count':
                        res[_pos] = res[_pos] + _count
                    elif feature.metric == 'min':
                        res[_pos] = min(res[_pos], _min)
                    elif feature.metric == 'max':
                        res[_pos] = max(res[_pos], _max)
                    elif feature.metric == 'avg':
                        # avg computed in the end
                        res[_pos] = res[_pos] + _sum
                    elif feature.metric == 'sum':
                        res[_pos] = res[_pos] + _sum
                    elif feature.metric == 'stddev':
                        # std computed in the end
                        res[_pos] = res[_pos] + _sum_of_squares

        for quad_num in range(self.nb_quadrants):
            for feat_num, feature in enumerate(agg.features):
                quad_pos = quad_num * len(agg.features)
                _pos = quad_pos + feat_num
                if feature.metric == 'min' and res[_pos] == _MAX_INT:
                    res[_pos] = 0

            for feat_num, feature in enumerate(agg.features):
                _pos = quad_pos + feat_num
                _count = counts[_pos]
                _sum = sums[_pos]
                _sum_of_squares = sum_of_squares[_pos]

                if _count > 0:
                    if feature.metric == 'avg':
                        res[_pos] = _sum / _count
                    elif feature.metric == 'stddev':
                        _variance = math.sqrt(_sum_of_squares / _count - (_sum/_count) ** 2)
                        res[_pos] = _variance

        return res

    def _train_on_dataset(
        self,
        dataset,
        num_epochs=100,
        limit=-1,
        progress_cb=None,
    ):
        # Apply data standardization to each feature individually
        # https://en.wikipedia.org/wiki/Feature_scaling
        self._means = np.mean(dataset, axis=0)
        self._stds = np.std(dataset, axis=0)
        # force std=1.0 (normal distribution) if std is null
        self._stds[self._stds == 0] = 1.0
        zY = np.nan_to_num((dataset - self._means) / self._stds)

        # Hyperparameters
        data_dimens = self.nb_dimensions
        self._som_model = som.SOM(self.w, self.h, data_dimens, num_epochs)

        # Start Training
        self._som_model.train(zY, truncate=limit, progress_cb=progress_cb)

        # Map vectors to their closest neurons
        return self._som_model.map_vects(zY)

    def set_fingerprint(
        self,
        key,
        fp,
    ):
        fps = {
            fingerprint['key']: fingerprint
            for fingerprint in self._state['fingerprints']
        }
        fps[key] = fp
        self._state['fingerprints'] = [val for key, val in fps.items()]
 
    def add_fingerprint(
        self,
        fp,
    ):
        fp['_fingerprint'] = [0] * len(fp['_fingerprint'])
        fp['fingerprint'] = [0] * len(fp['fingerprint'])
        self._state['fingerprints'].append(fp)

    def _norm_features(
        self,
        x,
    ):
        return np.nan_to_num((x - self._means) / self._stds)

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
            _fingerprint = self._norm_features(dataset[i])
            fingerprints.append({
                'key': key,
                'time_range': (int(from_ts), int(to_ts)),
                'fingerprint': dataset[i].tolist(),
                '_fingerprint': _fingerprint.tolist(),
                'location': (mapped[i][0].item(), mapped[i][1].item()),
            })

        return fingerprints

    def _get_norm_mul(self, from_ts, to_ts):
        if self.is_trained:
            training_from_ts = make_ts(self._state['from_date'])
            training_to_ts = make_ts(self._state['to_date'])
        else:
            training_from_ts = from_ts
            training_to_ts = to_ts

        training_time_range = training_to_ts - training_from_ts
        time_range = to_ts - from_ts
        norm_mul = np.full((self.nb_quadrants, self.nb_features), np.nan, dtype=float)
        for quad_num in range(self.nb_quadrants):
            for i, feature in enumerate(self.features):
                if feature.metric == 'count':
                    norm_mul[quad_num, i] = training_time_range / time_range
                elif feature.metric == 'sum':
                    norm_mul[quad_num, i] = training_time_range / time_range
                else:
                    norm_mul[quad_num, i] = 1.0

        return np.ravel(norm_mul)

    def _get_low_high(self):
        dimens = self.nb_dimensions
        low = np.full((dimens,), np.nan, dtype=float)
        high = np.full((dimens,), np.nan, dtype=float)
        for quad_num in range(self.nb_quadrants):
            for feat_num, feature in enumerate(self.features):
                quad_pos = quad_num * len(self.features)
                _pos = quad_pos + feat_num
                if feature.low_watermark is not None:
                    low[_pos] = feature.low_watermark
                if feature.high_watermark is not None:
                    high[_pos] = feature.high_watermark

        return low, high

    def _make_dataset(self, dicts, from_ts, to_ts):
        keys = set()
        for d in dicts:
            keys = keys.union(d.keys())

        nb_keys = len(keys)
        dimens = self.nb_dimensions
        dataset = np.zeros((nb_keys, dimens), dtype=float)

        low, high = self._get_low_high()
        mul = self._get_norm_mul(from_ts, to_ts)

        for i, key in enumerate(keys):
            col = 0
            row = np.zeros((1, dimens), dtype=float)

            for agg_num, agg in enumerate(self.aggs):
                features_len = len(agg.features)
                if key in dicts[agg_num]:
                    features = dicts[agg_num][key]
                    for quad_num in range(self.nb_quadrants):
                        quad_pos = quad_num * features_len
                        row_pos = quad_num * self.nb_features + col
                        row[0][row_pos:row_pos+features_len] = features[quad_pos:quad_pos+features_len]
                col = col + features_len

            row[0] *= mul
            row[0] = np.nanmax([row[0], low], axis=0)
            row[0] = np.nanmin([row[0], high], axis=0)

            dataset[i] = row

        return list(keys), dataset

    def train(
        self,
        datasource,
        from_date,
        to_date="now",
        num_epochs=100,
        limit=-1,
        progress_cb=None,
        license=None,
    ):
        self._som_model = None
        self._means = None
        self._stds = None

        self.check_allowed_date_range(from_date, to_date, license)

        from_ts = make_ts(from_date)
        to_ts = make_ts(to_date)

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

        # Fill dataset
        features_dicts=[]
        for agg_num, agg in enumerate(self.aggs):
            data = datasource.get_quadrant_data(self, agg, from_ts, to_ts)
            features = dict()
            for key, val in data:
                features[key] = self.format_quadrants(val, agg)
            features_dicts.append(features)

        keys, dataset = self._make_dataset(features_dicts, from_ts, to_ts)

        if len(keys) == 0:
            raise errors.NoData("no data found for time range {}-{}".format(
                from_str,
                to_str,
            ))

        logging.info("found %d keys", len(keys))

        mapped = self._train_on_dataset(
            dataset,
            num_epochs,
            limit,
            progress_cb,
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
            'from_date': ts_to_str(from_ts),
            'to_date': ts_to_str(to_ts),
        }

    def unload(self):
        del self._som_model
        self._som_model = None

    def load(self):
        if not self.is_trained:
            return errors.ModelNotTrained()

        # exit if already loaded
        if self._som_model:
            return

        self._som_model = som.load_model(
            self._state['ckpt'],
            self._state['index'],
            self._state['meta'],
            self.w,
            self.h,
            self.nb_dimensions,
        )

    @property
    def preview(self):
        trained = self.is_trained

        state = {
            'trained': self.is_trained
        }

        return {
            'settings': self.settings,
            'state': state,
        }

    def _map_dataset(self, dataset, from_ts, to_ts):
        zY = self._norm_features(dataset)
        mapped = self._som_model.map_vects(zY)
        return mapped

    def predict(
        self,
        datasource,
        from_date,
        to_date,
        key_val=None,
        license=None,
    ):
        self.check_allowed_date_range(from_date, to_date, license)

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

        # Fill dataset
        features_dicts=[]
        for agg_num, agg in enumerate(self.aggs):
            data = datasource.get_quadrant_data(self, agg, from_ts, to_ts, key_val)
            features = dict()
            for key, val in data:
                features[key] = self.format_quadrants(val, agg)
            features_dicts.append(features)

        keys, dataset = self._make_dataset(features_dicts, from_ts, to_ts)

        if len(keys) == 0:
            raise errors.NoData("no data found for time range {}-{}".format(
                from_str,
                to_str,
            ))

        logging.info("found %d keys", len(keys))

        mapped = self._map_dataset(
            dataset,
            from_ts,
            to_ts,
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


    def predict_ranges(
        self,
        datasource,
        date_ranges,
        key_val=None,
    ):
        self.load()

        for from_date, to_date in date_ranges:
            from_ts = make_ts(from_date)
            to_ts = make_ts(to_date)
    
            # Fixup range to be sure that it is a multiple of interval
            from_ts = math.floor(from_ts / self.interval) * self.interval
            to_ts = math.ceil(to_ts / self.interval) * self.interval
    
            from_str = ts_to_str(from_ts)
            to_str = ts_to_str(to_ts)
    
            logging.info("predict(%s) range=[%s, %s]",
                         self.name, from_str, to_str)
    
            # Fill dataset
            features_dicts=[]
            for agg_num, agg in enumerate(self.aggs):
                data = datasource.get_quadrant_data(self, agg, from_ts, to_ts, key_val)
                data = {key:val for key, val in data}

                features = dict()
                for key, val in data.items():
                    features[key] = self.format_quadrants(val, agg)
                features_dicts.append(features)
    
            keys, dataset = self._make_dataset(features_dicts, from_ts, to_ts)
            if len(keys) == 0:
                logging.warning(errors.NoData("no data found for time range {}-{}".format(
                    from_str,
                    to_str,
                )))
                yield FingerprintsPrediction(
                    from_ts=from_ts,
                    to_ts=to_ts,
                    fingerprints=[],
                )
                continue
    
            logging.info("found %d keys", len(keys))
    
            mapped = self._map_dataset(
                dataset,
                from_ts,
                to_ts,
            )
    
            fingerprints = self._build_fingerprints(
                dataset,
                mapped,
                keys,
                from_ts,
                to_ts,
            )
    
            yield FingerprintsPrediction(
                from_ts=from_ts,
                to_ts=to_ts,
                fingerprints=fingerprints,
            )

    def predict_ranges_and_scores(
        self,
        datasource,
        date_ranges,
        key_val=None,
        cpu_count=os.cpu_count(),
    ):
        pool = Pool()
        for _date_ranges in chunks(date_ranges, size=cpu_count):
            local_ranges = list(_date_ranges)
            local_args = zip(repeat(self, len(local_ranges)), \
                             repeat(datasource, len(local_ranges)), \
                             repeat(key_val, len(local_ranges)), \
                             local_ranges)
            res = pool.map(predict_scores, local_args)
            for prediction in sorted(res, key=lambda x: x.from_ts):
                yield prediction

        pool.close()

    def show(self, show_summary=False):
        exn = self.load()
        if exn:
            raise(exn)

        som_model = self._som_model
        fingerprints = self._state['fingerprints']
        centroids = som_model.centroids()
        result = {
            'fingerprints': fingerprints
        }
        counts = np.zeros(shape=(self.h, self.w), dtype=int)
        locations = {}
        for fingerprint in fingerprints:
            x, y = fingerprint['location']
            counts[x,y] += 1
            l = locations.get((x,y)) or []
            l.append(fingerprint['key'])
            locations[x,y] = l

        if show_summary == True:
            l = [ [x, y, locs] for (x, y), locs in locations.items() ]
            detail = '\n'.join('{},{}: {}'.format(x, y, loc) for x, y, loc in sorted(l, key = operator.itemgetter(0, 1)))
            grid = '\n'.join([''.join(['{:3}'.format(cnt) for cnt in row]) for row in counts])
            return grid + '\n' + detail

        return result

    def generate_fake_prediction(self):
        #to_ts = datetime.datetime.now().timestamp()
        to_ts = make_ts('2017-12-25T04:00:00.000Z')
        from_ts = to_ts - self.span

        training_from_ts = make_ts(self._state['from_date'])
        training_to_ts = make_ts(self._state['to_date'])

        key = '1018797' # FIXME: should be an input argument
        dimens = self.nb_dimensions
        sigma = 5
        fingerprint = self._means + sigma * self._stds
        _fingerprint = np.zeros(shape=(1,dimens), dtype=float)
        _fingerprint[:] = sigma
        mapped = self._som_model.map_vects(_fingerprint)
        _location = (mapped[0][0].item(), mapped[0][1].item())

        fingerprints = [
            {
                "fingerprint": fingerprint.tolist(),
                "_fingerprint": _fingerprint[0].tolist(),
                "location": _location,
                "key": key,
                "time_range": [
                    int(training_from_ts),
                    int(training_to_ts)
                ],
            }
        ]

        return FingerprintsPrediction(
            from_ts=from_ts,
            to_ts=to_ts,
            fingerprints=fingerprints,
        )

    def detect_anomalies(self, prediction, hooks=[]):
        """
        Detect anomalies on observed data by comparing them to the values
        predicted by the model
        """

        self.load()

        fps = {
            fingerprint['key']: fingerprint
            for fingerprint in self._state['fingerprints']
        }

        prediction.changed = []
        prediction.anomalies = None

        low_highs = [feature.anomaly_type for feature in self.features]

        dimens = self.nb_dimensions
        _fingerprint = np.zeros(shape=(1, dimens), dtype=float)
        mapped = self._som_model.map_vects(_fingerprint)
        _location = (mapped[0][0].item(), mapped[0][1].item())

        dt = ts_to_datetime(prediction.to_ts)
        date_str = datetime_to_str(dt)
        for fp_pred in prediction.fingerprints:
            is_anomaly = False
            anomalies = {}

            key = fp_pred['key']
            fp = fps.get(key)

            if fp is None:
                # signature = initial. We haven't seen this key during training
                prediction.changed.append(key)
                # Assign zeros ie, an all-average profile by default
                fp = {}
                fp['fingerprint'] = self._means
                fp['_fingerprint'] = _fingerprint[0].tolist()
                fp['location'] = _location

            scores = self._som_model.get_scores(
                np.array(fp['_fingerprint']),
                np.array(fp_pred['_fingerprint']),
                low_highs,
            )
            logging.info("scores for {} = {}".format(key, scores))
            max_arg = np.nanargmax(scores)
            max_score = scores[max_arg].item()

            for j, score in enumerate(scores):
                quadrant = int(j / self.nb_features)
                pos = int(j % self.nb_features)
                feature = self.features[pos]

                if score < self.max_threshold:
                    continue

                anomalies[feature.name] = {
                    'type': feature.anomaly_type,
                    'score': score,
                    'quadrant': quadrant,
                }

            if len(anomalies):
                is_anomaly = True

            stats = {
                'scores': scores.tolist(),
                'score': max_score,
                'anomaly': is_anomaly,
                'anomalies': anomalies,
            }

            if self._state.get('anomaly') is None:
                self._state['anomaly'] = {}

            anomaly = self._state.get('anomaly').get(key)

            if anomaly is None:
                if is_anomaly:
                    # This is a new anomaly

                    # TODO have a Model.logger to prefix all logs with model name
                    logging.warning("detected anomaly for model '%s' and key '%s' at %s (score = %.1f)",
                                    self.name, key, date_str, max_score)

                    self._state['anomaly'][key] = {
                        'start_ts': make_ts(date_str),
                        'max_score': max_score,
                    }

                    for hook in hooks:
                        logging.debug("notifying '%s' hook", hook.name)
                        hook.on_anomaly_start(
                            self.name,
                            dt=dt,
                            score=max_score,
                            predicted=fp_pred,
                            observed=None,
                            expected=fp,
                            key=key,
                            anomalies=anomalies,
                        )
            else:
                if is_anomaly:
                    anomaly['max_score'] = max(anomaly['max_score'], max_score)
                    logging.warning(
                        "anomaly still in progress for model '%s' and key '%s' at %s (score = %.1f)",
                        self.name, key, date_str, max_score,
                    )
                    self._state['anomaly'][key] = anomaly
                elif score < self.min_threshold:
                    logging.info(
                        "anomaly ended for model '%s' and key '%s' at %s (score = %.1f)",
                        self.name, key, date_str, max_score,
                    )

                    for hook in hooks:
                        logging.debug("notifying '%s' hook", hook.name)
                        hook.on_anomaly_end(self.name, dt, max_score, key=key)

                    self._state['anomaly'].pop(key)

            fp_pred['stats'] = stats

        prediction.anomalies = self._state['anomaly']

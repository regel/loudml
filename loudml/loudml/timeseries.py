"""
LoudML time-series module
"""

import datetime
import json
import logging
import math
import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.layers import Activation
from tensorflow.contrib.keras.api.keras.layers import LSTM
from tensorflow.contrib.keras.api.keras.callbacks import EarlyStopping

from hyperopt import hp
from hyperopt import space_eval
from hyperopt import (
    fmin,
    STATUS_OK,
    STATUS_FAIL,
    tpe,
    Trials,
)

from voluptuous import (
    All,
    Any,
    Boolean,
    Required,
    Optional,
    Range,
    Schema,
)

from . import (
    errors,
    schemas,
)
from .misc import (
    datetime_to_str,
    dt_get_weekday,
    dt_get_daytime,
    make_datetime,
    make_ts,
    parse_timedelta,
    ts_to_str,
    ts_to_datetime,
)
from .model import (
    Model,
)

DEFAULT_SEASONALITY = {
    'daytime': False,
    'weekday': False,
}

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

# global vars for easy reusability
# This UNIX process is handling a unique model
_keras_model, _graph = None, None
_verbose = 0

# _hp_forecast_min=1 Backward compat with 1.2
_hp_forecast_min = 1
_hp_forecast_max = 5
_hp_span_min = 5
_hp_span_max = 20

def _transform(feature, y):
    if feature.transform == "diff":
        return np.concatenate(([np.nan], np.diff(y, axis=0)))

def _revert_transform(feature, y, y0):
    if feature.transform == "diff":
        return np.cumsum(np.concatenate(([y0], y[1:])))

def canonicalize_min_max(value, _min, _max):
    return 1.0 - (_max - value) / (_max - _min)

def uncanonicalize_min_max(value, _min, _max):
    return _max - (_max - _min) * (1.0 - value)

def _get_scores(feature, y, _min, _max, _mean, _std):
    if feature.scores == "min_max":
        y = canonicalize_min_max(y, _min, _max)
    elif feature.scores == "normalize":
        y0 = y[~np.isnan(y)][0]
        if y0 > 0 and y0 < 0.01:
            y0 = 0.01
        elif y0 < 0 and y0 > -0.01:
            y0 = -0.01
        y = (y / y0) - 1.0
    elif feature.scores == "standardize":
        y = (y - _mean) / _std
    return y

def _revert_scores(feature, y, _data, _min, _max, _mean, _std):
    if feature.scores == "min_max":
        y = uncanonicalize_min_max(y, _min, _max)
    elif feature.scores == "normalize":
        p0 = _data[~np.isnan(_data)][0]
        y = p0 * (y + 1.0)
    elif feature.scores == "standardize":
        y = (y * _std) + _mean
    return y


class HyperParameters:
    """Hyperparameters"""

    def __init__(self, params=None):
        if params:
            self.assign(params)

    def assign(self, params):
        """
        Assign hyperparameters
        """

        for key, value in params.items():
            try:
                if int(value) == value:
                    value = int(value)
                elif float(value) == value:
                    value = float(value)
            except ValueError:
                pass
            setattr(self, key, value)

def _serialize_keras_model(keras_model):
    """
    Serialize Keras model
    """

    import base64
    import tempfile
    import h5py

    model_b64 = base64.b64encode(keras_model.to_json().encode('utf-8'))

    fd, path = tempfile.mkstemp()
    try:
        keras_model.save_weights(path)
        with os.fdopen(fd, 'rb') as tmp:
            weights_b64 = base64.b64encode(tmp.read())
    finally:
        os.remove(path)

    return model_b64.decode('utf-8'), weights_b64.decode('utf-8')

def _load_keras_model(model_b64, weights_b64, loss_fct, optimizer):
    """
    Load Keras model
    """
    import tempfile
    import base64
    import h5py
    # Note: the import were moved here to avoid the speed penalty
    # in code that imports the storage module
    import tensorflow as tf
    import tensorflow.contrib.keras.api.keras.models
    from tensorflow.contrib.keras.api.keras.models import model_from_json

    model_json = base64.b64decode(model_b64.encode('utf-8')).decode('utf-8')
    keras_model = model_from_json(model_json)

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(base64.b64decode(weights_b64.encode('utf-8')))
            tmp.close()
    finally:
        # load weights into new model
        keras_model.load_weights(path)
        os.remove(path)

    keras_model.compile(loss=loss_fct, optimizer=optimizer)
    graph = tf.get_default_graph()

    return keras_model, graph

class TimeSeriesPrediction:
    """
    Time-series prediction
    """

    def __init__(self, model, timestamps, observed, predicted):
        self.model = model
        self.timestamps = timestamps
        self.observed = observed
        self.predicted = predicted
        self.anomaly_indices = None
        self.stats = None

    def get_anomalies(self):
        """
        Return anomalies
        """

        if self.anomaly_indices is None:
            raise errors.NotFound("anomaly detection has not been performed yet")
        return [self._format_bucket(i) for i in self.anomaly_indices]

    def apply_default(self, feature_idx, value):
        """
        Apply default feature value
        """
        if value is None or value is np.nan or np.isnan(value):
            value = self.model.defaults[feature_idx]
            if value is None or value is np.nan or np.isnan(value):
                return None
            else:
                return value

        return value

    def format_series(self):
        """
        Return prediction data as a time-series
        """

        observed = {}
        predicted = {}

        for i, feature in enumerate(self.model.features):
            if feature.is_input:
                observed[feature.name] = [self.apply_default(i, x) for x in self.observed[:,i]]
            if feature.is_output:
                predicted[feature.name] = [self.apply_default(i, x) for x in self.predicted[:,i]]

        result = {
            'timestamps': self.timestamps,
            'observed': observed,
            'predicted': predicted,
        }
        if self.stats is not None:
            result['stats'] = self.stats,
        return result

    def format_bucket_data(self, i):
        """
        Format observation and prediction for one bucket
        """
        features = self.model.features
        return {
            'observed': {
                feature.name: self.apply_default(j, self.observed[i][j])
                for j, feature in enumerate(features) if feature.is_input==True
            },
            'predicted': {
                feature.name: self.apply_default(j, self.predicted[i][j])
                for j, feature in enumerate(features) if feature.is_output==True
            },
        }

    def _format_bucket(self, i):
        """
        Format one bucket
        """

        bucket = self.format_bucket_data(i)
        bucket['timestamp'] = self.timestamps[i]
        if self.stats:
            bucket['stats'] = self.stats[i]
        return bucket

    def format_buckets(self):
        """
        Return prediction data as buckets
        """

        return [
            self._format_bucket(i)
            for i, _ in enumerate(self.timestamps)
        ]

    def __str__(self):
        return json.dumps(self.format_buckets(), indent=4)

    def plot(self, feature_name):
        """
        Plot prediction
        """

        import matplotlib.pylab as plt

        i = None
        for i, feature in enumerate(self.model.features):
            if feature.name == feature_name:
                break

        if i is None:
            raise errors.NotFound("feature not found")

        plt.rcParams["figure.figsize"] = (17, 9)
        plt.plot(self.observed[:,i],"--")
        plt.plot(self.predicted[:,i],":")
        plt.show()


class TimeSeriesModel(Model):
    """
    Time-series model
    """
    TYPE = 'timeseries'

    SCHEMA = Model.SCHEMA.extend({
        Required('bucket_interval'): schemas.TimeDelta(
            min=0, min_included=False,
        ),
        Required('interval'): schemas.TimeDelta(min=0, min_included=False),
        Required('offset'): schemas.TimeDelta(min=0),
        Required('span'): Any(None, "auto", All(int, Range(min=1))),
        Optional('min_span'): All(int, Range(min=1)),
        Optional('max_span'): All(int, Range(min=1)),
        Optional('seasonality', default=DEFAULT_SEASONALITY): schemas.seasonality,
        Optional('forecast'): Any(None, "auto", All(int, Range(min=1))),
        Optional('min_forecast'): All(int, Range(min=1)),
        Optional('max_forecast'): All(int, Range(min=1)),
        'timestamp_field': schemas.key,
    })

    def __init__(self, settings, state=None):
        global _hp_span_min, _hp_span_max
        global _hp_forecast_min, _hp_forecast_max
        super().__init__(settings, state)

        self.timestamp_field = settings.get('timestamp_field', 'timestamp')
        self.bucket_interval = parse_timedelta(settings.get('bucket_interval')).total_seconds()
        self.interval = parse_timedelta(settings.get('interval')).total_seconds()
        self.offset = parse_timedelta(settings.get('offset')).total_seconds()

        self.span = settings.get('span')

        self.mins = None
        self.maxs = None
        self.means = None
        self.stds = None

        if self.span is None or self.span == "auto":
            self.min_span = settings.get('min_span') or _hp_span_min
            self.max_span = settings.get('max_span') or _hp_span_max
        else:
            self.min_span = self.span
            self.max_span = self.span

        self.forecast_val = settings.get('forecast') or _hp_forecast_min
        if self.forecast_val == "auto":
            self.min_forecast = settings.get('min_forecast') or _hp_forecast_min
            self.max_forecast = settings.get('max_forecast') or _hp_forecast_max
        else:
            self.min_forecast = self.forecast_val
            self.max_forecast = self.forecast_val

        self.sequential = None

        self.defaults = [
            np.nan if feature.default is np.nan else feature.default
            for feature in self.features
        ]

        self.current_eval = None

    def enum_features(self, is_input=None, is_output=None):
        j = 0
        for i, feature in enumerate(self.features):
            if feature.is_input == is_input or feature.is_output == is_output:
                yield i, j, feature
                j += 1

    @property
    def type(self):
        return self.TYPE

    def get_hp_span(self, label):
        if (self.max_span - self.min_span) <= 0:
            space = self.span
        else:
            space = self.min_span + hp.randint(label, (self.max_span - self.min_span))
        return space

    def get_hp_forecast(self, label):
        if (self.max_forecast - self.min_forecast) <= 0:
            space = self.forecast_val
        else:
            space = self.min_forecast + hp.randint(label, (self.max_forecast - self.min_forecast))
        return space

    def set_run_params(self, params=None):
        """
        Set running parameters to make them persistent
        """
        if params is None:
            self._settings.pop('run', None)
        else:
            self._settings['run'] = params

    def _compute_nb_buckets(self, from_ts, to_ts):
        """
        Compute the number of bucket between `from_ts` and `to_ts`
        """
        return int((to_ts - from_ts) / self.bucket_interval) + 2

    def apply_defaults(self, x):
        """
        Apply default feature value to np array
        """
        for i, feature in enumerate(self.features):
            x[np.isnan(x[:,i])] = self.defaults[i]

    def canonicalize_dataset(
        self,
        dataset,
        only_outputs=False,
        out=None,
    ):
        """
        Canonicalize dataset values
        """

        if out is None:
            out = np.empty_like(dataset)

        for i, j, feature in self.enum_features(
            is_input=not only_outputs,
            is_output=True,
        ):
            out[:,j] = _get_scores(
                feature,
                dataset[:,j],
                _min=self.mins[i],
                _max=self.maxs[i],
                _mean=self.means[i],
                _std=self.stds[i],
            )

        if only_outputs:
            return out

        if self.seasonality.get('daytime'):
            i += 1
            j += 1
            out[:,j] = canonicalize_min_max(dataset[:,i], 0, 23)

        if self.seasonality.get('weekday'):
            i += 1
            j += 1
            out[:,j] = canonicalize_min_max(dataset[:,i], 0, 6)

        return out

    def uncanonicalize_dataset(
        self,
        dataset,
        only_outputs=False,
        out=None,
    ):
        """
        Uncanonicalize dataset values
        """

        if out is None:
            out = np.empty_like(dataset)

        for i, j, feature in self.enum_features(
            is_input=not only_outputs,
            is_output=True,
        ):
            out[:,j] = _revert_scores(
                feature,
                dataset[:,j],
                _data=dataset[self._span:,i],
                _min=self.mins[i],
                _max=self.maxs[i],
                _mean=self.means[i],
                _std=self.stds[i],
            )

        if only_outputs:
            return out

        if self.seasonality.get('daytime'):
            i += 1
            j += 1
            out[:,j] = uncanonicalize_min_max(dataset[:,i], 0, 23)

        if self.seasonality.get('weekday'):
            i += 1
            j += 1
            out[:,j] = uncanonicalize_min_max(dataset[:,i], 0, 6)

        return out

    def stat_dataset(self, dataset):
        """
        Compute dataset sets and keep them as reference for canonicalization
        """
        self.mins = np.min(np.nan_to_num(dataset), axis=0)
        self.maxs = np.max(np.nan_to_num(dataset), axis=0)
        self.means = np.nanmean(dataset, axis=0)
        self.stds = np.nanstd(dataset, axis=0)

    def _train_on_dataset(
        self,
        dataset,
        train_size=0.67,
        batch_size=64,
        num_epochs=100,
        max_evals=None,
        progress_cb=None,
    ):
        if max_evals is None:
            max_evals = self.settings.get('max_evals', 10)

        self.current_eval = 0

        self.stat_dataset(dataset)

        # Canonicalize dataset in-place
        self.canonicalize_dataset(dataset, out=dataset)

        input_features = len(self._x_indexes())

        logging.info("Preprocessing. mins: %s maxs: %s ranges: %s",
                     self.mins, self.maxs, self.maxs - self.mins)

        def cross_val_model(params):
            global _keras_model, _graph
            _keras_model, _graph = None, None
            # Destroys the current TF graph and creates a new one.
            # Useful to avoid clutter from old models / layers.
            K.clear_session()

            self.span = params.span
            self.forecast_val = params.forecast
            (_, X_train, y_train), (_, X_test, y_test) = self.train_test_split(
                dataset,
                train_size=train_size,
            )

            # expected input data shape: (batch_size, timesteps, nb_features)
            _keras_model = Sequential()
            if params.depth == 1:
                _keras_model.add(LSTM(
                    params.l1,
                    input_shape=(None, input_features),
                    return_sequences=False,
                ))
                _keras_model.add(Dense(y_train.shape[1], input_dim=params.l1))
            elif params.depth == 2:
                _keras_model.add(LSTM(
                    params.l1,
                    input_shape=(None, input_features),
                    return_sequences=True,
                ))
                _keras_model.add(LSTM(params.l2, return_sequences=False))
                _keras_model.add(Dense(y_train.shape[1], input_dim=params.l2))

            _keras_model.add(Activation(params.activation))
            _keras_model.compile(
                loss=params.loss_fct,
                optimizer=params.optimizer,
                metrics=['accuracy'],
            )
            _stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                verbose=_verbose,
                mode='auto',
            )
            _keras_model.fit(
                X_train,
                y_train,
                epochs=num_epochs,
                batch_size=batch_size,
                verbose=_verbose,
                validation_data=(X_test, y_test),
                callbacks=[_stop],
            )

            # How well did it do?
            scores = _keras_model.evaluate(
                X_test,
                y_test,
                batch_size=batch_size,
                verbose=_verbose,
            )

            self.current_eval += 1
            if progress_cb is not None:
                progress_cb(self.current_eval, max_evals)

            return scores

        hyperparameters = HyperParameters()

        # Parameter search space
        def objective(args):
            hyperparameters.assign(args)

            try:
                score = cross_val_model(hyperparameters)
                return {'loss': score[0], 'status': STATUS_OK}
            except Exception as exn:
                logging.warning("iteration failed: %s", exn)
                return {'loss': None, 'status': STATUS_FAIL}

        space = hp.choice('case', [
            {
              'depth': 1,
              'span': self.get_hp_span('d1_span'),
              'forecast': self.get_hp_forecast('d1_forecast'),
              'l1': 1+hp.randint('d1_l1', 100),
              'activation': hp.choice('d1_activation', ['linear', 'tanh']),
              'loss_fct': hp.choice('d1_loss_fct', ['mean_squared_error']),
              'optimizer': hp.choice('d1_optimizer', ['adam']),
            },
            {
              'depth': 2,
              'span': self.get_hp_span('d2_span'),
              'forecast': self.get_hp_forecast('d2_forecast'),
              'l1': 1+hp.randint('d2_l1', 100),
              'l2': 1+hp.randint('d2_l2', 100),
              'activation': hp.choice('d2_activation', ['linear', 'tanh']),
              'loss_fct': hp.choice('d2_loss_fct', ['mean_squared_error']),
              'optimizer': hp.choice('d2_optimizer', ['adam']),
            }
        ])

        # The Trials object will store details of each iteration
        trials = Trials()

        # Run the hyperparameter search using the tpe algorithm
        best = fmin(
            objective,
            space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
        )

        # Get the values of the optimal parameters
        best_params = space_eval(space, best)
        score = cross_val_model(HyperParameters(best_params))
        self.span = best_params['span']
        self.forecast_val = best_params['forecast']
        (_, X_train, y_train), (_, X_test, y_test) = self.train_test_split(
            dataset,
            train_size=train_size,
        )

        predicted = _keras_model.predict(X_test)
        return (best_params, score, y_test[:], predicted[:])

    def _format_dataset(self, dataset):
        """
        Format dataset for time-series training

        It is assumed that a value for a given bucket can be predicted
        according the preceding ones. The number of preceding buckets used
        for prediction is given by `self.span`.

        input:
        [v0, v1, v2, v3, v4 ..., vn]

        output:
        indexes = [3, 4, ..., n]
        X = [
            [v0, v1, v2], # span = 3
            [v1, v2, v3],
            [v2, v3, v4],
            ...
            [..., .., vn],
        ]
        y = [
            v3,
            v4,
            ...
            vn,
        ]

        Buckets with missing values are skipped.
        """
        data_x, data_y = [], []
        indexes = []

        x_indexes = self._x_indexes()
        y_indexes = self._y_indexes()
        for i in range(len(dataset) - self.span - self.forecast_val + 1):
            j = i + self.span
            k = i + self.span + self.forecast_val
            partX = dataset[i:j, x_indexes]
            partY = np.ravel(dataset[j:k, y_indexes])

            if not np.isnan(partX).any() and not np.isnan(partY).any():
                data_x.append(partX)
                data_y.append(partY)
                indexes.append(j)

        return np.array(indexes), np.array(data_x), np.array(data_y)

    def train_test_split(self, dataset, train_size=0.67):
        """
        Splits data to training and testing parts
        """

        ntrn = round(len(dataset) * train_size)
        i_sel, X_train, y_train = self._format_dataset(dataset[0:ntrn])
        j_sel, X_test, y_test = self._format_dataset(dataset[ntrn:])
        return (i_sel, X_train, y_train), (j_sel, X_test, y_test)

    def train(
        self,
        datasource,
        from_date,
        to_date="now",
        train_size=0.67,
        batch_size=64,
        num_epochs=100,
        max_evals=None,
        progress_cb=None,
    ):
        """
        Train model
        """
        global _keras_model, _graph

        _keras_model, _graph = None, None

        self.mins, self.maxs = None, None
        self.means, self.stds = None, None

        from_ts = make_ts(from_date)
        to_ts = make_ts(to_date)

        from_str = ts_to_str(from_ts)
        to_str = ts_to_str(to_ts)

        logging.info(
            "train(%s) range=[%s, %s] train_size=%f batch_size=%d epochs=%d)",
            self.name,
            from_str,
            to_str,
            train_size,
            batch_size,
            num_epochs,
        )

        # Prepare dataset
        nb_buckets = self._compute_nb_buckets(from_ts, to_ts)
        nb_features = len(self.features)
        dataset = np.zeros((nb_buckets, nb_features), dtype=float)
        dataset[:] = np.nan
        daytime = np.zeros((nb_buckets, 1), dtype=float)
        weekday = np.zeros((nb_buckets, 1), dtype=float)

        # Fill dataset
        data = datasource.get_times_data(self, from_ts, to_ts)

        i = None
        for i, (_, val, timeval) in enumerate(data):
            dataset[i] = val

            dt = make_datetime(timeval)
            daytime[i] = np.array(dt_get_daytime(dt))
            weekday[i] = np.array(dt_get_weekday(dt))

        self.apply_defaults(dataset)

        if i is None:
            raise errors.NoData("no data found for time range {}-{}".format(
                from_str,
                to_str,
            ))

        logging.info("found %d time periods", i + 1)

        for j, feature in enumerate(self.features):
            if feature.transform is not None:
                dataset[:,j] = _transform(feature, dataset[:,j])

        if self.seasonality.get('daytime'):
            dataset = np.append(dataset, daytime, axis=1)
        if self.seasonality.get('weekday'):
            dataset = np.append(dataset, weekday, axis=1)

        best_params, score, _, _ = self._train_on_dataset(
            dataset,
            train_size,
            batch_size,
            num_epochs,
            max_evals,
            progress_cb=progress_cb,
        )
        self.current_eval = None

        for key, val in best_params.items():
            if not isinstance(val, str) and \
               not isinstance(val, int) and \
               not isinstance(val, float):
                best_params[key] = np.asscalar(val)

        model_b64, weights_b64 = _serialize_keras_model(_keras_model)

        self._state = {
            'graph': model_b64,
            'weights': weights_b64, # H5PY data encoded in base64
            'loss_fct': best_params['loss_fct'],
            'optimizer': best_params['optimizer'],
            'best_params': best_params,
            'mins': self.mins.tolist(),
            'maxs': self.maxs.tolist(),
            'means': self.means.tolist(),
            'stds': self.stds.tolist(),
            'loss': score[0],

        }

        return {
            'loss': score[0],
        }

    def load(self):
        """
        Load current model
        """
        global _keras_model, _graph

        if not self.is_trained:
            raise errors.ModelNotTrained()

        _keras_model, _graph = _load_keras_model(
            self._state['graph'],
            self._state['weights'],
            self._state['loss_fct'],
            self._state['optimizer'],
        )

        self.mins = np.array(self._state['mins'])
        self.maxs = np.array(self._state['maxs'])

        if 'means' in self._state:
            self.means = np.array(self._state['means'])
        if 'stds' in self._state:
            self.stds = np.array(self._state['stds'])

    @property
    def is_trained(self):
        """
        Tells if model is trained
        """
        return self._state is not None and 'weights' in self._state

    @property
    def _span(self):
        if 'span' in self._state['best_params']:
            return self._state['best_params']['span']
        else:
            return self.span

    @property
    def _forecast(self):
        if 'forecast' in self._state['best_params']:
            return self._state['best_params']['forecast']
        else:
            return self.forecast_val

    def _xy_indexes(self):
        """
        Return array of feature indices that are input and output for training and prediction
        """
        ii = set(self._x_indexes()) & set(self._y_indexes())
        return sorted(list(ii))

    def _y_indexes(self):
        """
        Return array of feature indices that are output for training and predictions
        """
        all_features = self.features
        y_indexes = []
        for index, feature in enumerate(all_features):
            if feature.is_output == True:
                y_indexes.append(index)

        return y_indexes

    def _x_indexes(self):
        """
        Return array of feature indices that must be input to training
        """
        all_features = self.features
        x_indexes = []
        for index, feature in enumerate(all_features):
            if feature.is_input == True:
                x_indexes.append(index)

        if self.seasonality.get('daytime'):
            index += 1
            x_indexes.append(index)
        if self.seasonality.get('weekday'):
            index += 1
            x_indexes.append(index)

        return x_indexes

    def _format_dataset_predict(self, dataset):
        """
        Format dataset for time-series prediction

        It is assumed that a value for a given bucket can be predicted
        according the preceding ones. The number of preceding buckets used
        for prediction is given by `self._span`.

        input:
        [v0, v1, v2, v3, v4 ..., vn]

        output:
        indexes = [3, 4, ..., n]
        X = [
            [v0, v1, v2], # _span = 3
            [v1, v2, v3],
            [v2, v3, v4],
            ...
            [..., .., vn],
        ]

        Buckets with missing values are skipped.
        """
        data_x = []
        indexes = []

        x_indexes = self._x_indexes()
        for i in range(len(dataset) - self._span + 1):
            j = i + self._span
            partX = dataset[i:j, x_indexes]

            if not np.isnan(partX).any():
                data_x.append(partX)
                indexes.append(j)

        return np.array(indexes), np.array(data_x)

    def predict(
        self,
        datasource,
        from_date,
        to_date,
    ):
        global _keras_model

        from_ts = make_ts(from_date)
        to_ts = make_ts(to_date)

        # Fixup range to be sure that it is a multiple of bucket_interval
        from_ts = math.floor(from_ts / self.bucket_interval) * self.bucket_interval
        to_ts = math.ceil(to_ts / self.bucket_interval) * self.bucket_interval

        from_str = ts_to_str(from_ts)
        to_str = ts_to_str(to_ts)

        # This is the number of buckets that the function MUST return
        predict_len = int((to_ts - from_ts) / self.bucket_interval)

        logging.info("predict(%s) range=[%s, %s]",
                     self.name, from_str, to_str)

        self.load()

        # Build history time range
        # Extra data are required to predict first buckets
        hist_from_ts = from_ts - self._span * self.bucket_interval
        hist_to_ts = to_ts
        hist_from_str = ts_to_str(hist_from_ts)
        hist_to_str = ts_to_str(hist_to_ts)

        # Prepare dataset
        nb_buckets = int((hist_to_ts - hist_from_ts) / self.bucket_interval)
        nb_features = len(self.features)
        dataset = np.empty((nb_buckets, nb_features), dtype=float)
        dataset[:] = np.nan
        daytime = np.zeros((nb_buckets, 1), dtype=float)
        weekday = np.zeros((nb_buckets, 1), dtype=float)

        X = []

        # Fill dataset
        logging.info("extracting data for range=[%s, %s]",
                     hist_from_str, hist_to_str)
        data = datasource.get_times_data(self, hist_from_ts, hist_to_ts)

        # Only a subset of history will be used for computing the prediction
        X_until = None # right bound for prediction
        i = None

        for i, (_, val, timeval) in enumerate(data):
            dataset[i] = val

            dt = make_datetime(timeval)
            daytime[i] = np.array(dt_get_daytime(dt))
            weekday[i] = np.array(dt_get_weekday(dt))

            ts = dt.timestamp()
            if ts < to_ts - self.bucket_interval:
                X.append(make_ts(timeval))
                X_until = i + 1

        self.apply_defaults(dataset)

        if i is None:
            raise errors.NoData("no data found for time range {}-{}".format(
                hist_from_str,
                hist_to_str,
            ))

        nb_buckets_found = i + 1
        if nb_buckets_found < nb_buckets:
            dataset = np.resize(dataset, (nb_buckets_found, nb_features))
            daytime = np.resize(daytime, (nb_buckets_found, 1))
            weekday = np.resize(weekday, (nb_buckets_found, 1))

        logging.info("found %d time periods", nb_buckets_found)

        real = np.copy(dataset)
        for j, feature in enumerate(self.features):
            if feature.transform is not None:
                dataset[:,j] = _transform(feature, dataset[:,j])

        if self.seasonality.get('daytime'):
            dataset = np.append(dataset, daytime, axis=1)
        if self.seasonality.get('weekday'):
            dataset = np.append(dataset, weekday, axis=1)

        # XXX For backward compatibility
        if self.means is None:
            logging.warning("model state has no mean values, new training needed")
            self.means = np.nanmean(dataset, axis=0)
        if self.stds is None:
            logging.warning("model state has no std values, new training needed")
            self.stds = np.nanstd(dataset, axis=0)

        norm_dataset = self.canonicalize_dataset(dataset)

        data_indexes, X_test = self._format_dataset_predict(norm_dataset[:X_until])

        if len(X_test) == 0:
            raise errors.LoudMLException("not enough data for prediction")

        logging.info("generating prediction")
        Y_ = _keras_model.predict(X_test).reshape((len(X_test), self._forecast, len(self._y_indexes())))[:,0,:]

        # FIXME: np.clip() according to feature preprocessing choice
        # Y_[Y_ < 0] = 0

        Y = self.uncanonicalize_dataset(Y_, only_outputs=True)

        for i, j, feature in self.enum_features(is_output=True):
            if feature.transform is not None:
                Y[:,j] = _revert_transform(
                    feature,
                    Y[:,j],
                    real[self._span,j],
                )

        # Build final result
        timestamps = X[self._span:]
        last_ts = make_ts(X[-1])
        timestamps.append(last_ts + self.bucket_interval)

        observed = np.array([self.defaults] * predict_len)
        predicted = np.array([self.defaults] * predict_len)

        for i, j, feature in self.enum_features(is_input=True):
            observed[:,i] = real[self._span:][:,i]
        for i, j, feature in self.enum_features(is_output=True):
            predicted[data_indexes - self._span,i] = Y[:][:,j]

        return TimeSeriesPrediction(
            self,
            timestamps=timestamps,
            observed=observed,
            predicted=predicted,
        )

    def generate_fake_prediction(self):
        now_ts = datetime.datetime.now().timestamp()
        timestamps = [
            now_ts - 2 * self.bucket_interval,
            now_ts - self.bucket_interval,
            now_ts,
        ]
        normal = [0.0] * len(self.features)
        anomaly = [sys.float_info.max] * len(self.features)

        return TimeSeriesPrediction(
            self,
            timestamps=timestamps,
            observed=np.array([normal, anomaly, normal]),
            predicted=np.array([normal, normal, normal]),
        )

    def forecast(
        self,
        datasource,
        from_date,
        to_date,
    ):
        global _keras_model

        from_ts = make_ts(from_date)
        to_ts = make_ts(to_date)

        # Fixup range to be sure that it is a multiple of bucket_interval
        from_ts = math.floor(from_ts / self.bucket_interval) * self.bucket_interval
        to_ts = math.ceil(to_ts / self.bucket_interval) * self.bucket_interval

        from_str = ts_to_str(from_ts)
        to_str = ts_to_str(to_ts)

        # This is the number of buckets that the function MUST return
        forecast_len = int((to_ts - from_ts) / self.bucket_interval)

        logging.info("forecast(%s) range=[%s, %s]",
                     self.name, from_str, to_str)

        self.load()

        # Build history time range
        # Extra data are required to forecast first buckets
        hist_from_ts = from_ts - self._span * self.bucket_interval
        hist_to_ts = from_ts
        hist_from_str = ts_to_str(hist_from_ts)
        hist_to_str = ts_to_str(hist_to_ts)

        # Prepare dataset
        nb_buckets = int((hist_to_ts - hist_from_ts) / self.bucket_interval)
        nb_features = len(self.features)
        y_indexes = self._y_indexes()
        nb_outputs = len(y_indexes)
        dataset = np.empty((nb_buckets, nb_features), dtype=float)
        dataset[:] = np.nan
        daytime = np.zeros((nb_buckets, 1), dtype=float)
        weekday = np.zeros((nb_buckets, 1), dtype=float)

        # Fill dataset
        logging.info("extracting data for range=[%s, %s]",
                     hist_from_str, hist_to_str)
        data = datasource.get_times_data(self, hist_from_ts, hist_to_ts)

        i = None
        for i, (_, val, timeval) in enumerate(data):
            dataset[i] = val
            dt = make_datetime(timeval)
            daytime[i] = np.array(dt_get_daytime(dt))
            weekday[i] = np.array(dt_get_weekday(dt))

        if i is None:
            raise errors.NoData("no data found for time range {}-{}".format(
                hist_from_str,
                hist_to_str,
            ))

        nb_buckets_found = i + 1
        if nb_buckets_found < nb_buckets:
            dataset = np.resize(dataset, (nb_buckets_found, nb_features))
            daytime = np.resize(daytime, (nb_buckets_found, 1))
            weekday = np.resize(weekday, (nb_buckets_found, 1))

        logging.info("found %d time periods", nb_buckets_found)

        for j, feature in enumerate(self.features):
            if feature.transform is not None:
                dataset[:,j] = _transform(feature, dataset[:,j])

        if self.seasonality.get('daytime'):
            dataset = np.append(dataset, daytime, axis=1)
        if self.seasonality.get('weekday'):
            dataset = np.append(dataset, weekday, axis=1)

        # XXX For backward compatibility
        if self.means is None:
            logging.warning("model state has no mean values, new training needed")
            self.means = np.nanmean(dataset, axis=0)
        if self.stds is None:
            logging.warning("model state has no std values, new training needed")
            self.stds = np.nanstd(dataset, axis=0)

        for j, feature in enumerate(self.features):
            dataset[:,j] = _get_scores(
                feature,
                dataset[:,j],
                _min=self.mins[j],
                _max=self.maxs[j],
                _mean=self.means[j],
                _std=self.stds[j],
            )

        data_indexes, X = self._format_dataset_predict(dataset)

        if len(X) == 0:
            raise errors.LoudMLException("not enough data for forecast")

        shape = (forecast_len, nb_features)
        predicted = np.array([self.defaults] * forecast_len)
        observed = np.empty(shape)
        observed[:] = np.nan

        logging.info("generating forecast")
        timestamps=[]
        bucket_start = from_ts
        bucket = 0
        xy_indexes = np.array(self._xy_indexes())
        while bucket_start < to_ts:
            Y_ = _keras_model.predict(X).reshape((self._forecast, nb_outputs))
            if len(xy_indexes) > 0:
                _xy_indexes = xy_indexes - np.min(xy_indexes)
                X[:, 0:self._forecast, xy_indexes] = Y_[:, _xy_indexes]
            if self.seasonality.get('daytime'):
                if self.seasonality.get('weekday'):
                    dt_pos = -2
                else:
                    dt_pos = -1

                dt_next = np.empty(shape=(self._forecast), dtype=float)
                dt_next[:] = np.nan
                for forecast_pos in range(self._forecast):
                    timeval = bucket_start + forecast_pos * self.bucket_interval
                    dt = make_datetime(timeval)
                    dt_next[forecast_pos] = dt_get_daytime(dt)

                max_value = self.maxs[dt_pos]
                rng = max_value - self.mins[dt_pos]

                norm_dt = 1.0 - (max_value - dt_next) / rng
                X[:, 0:self._forecast, dt_pos] = norm_dt

            if self.seasonality.get('weekday'):
                dt_pos = -1
                dt_next = np.empty(shape=(self._forecast), dtype=float)
                dt_next[:] = np.nan
                for forecast_pos in range(self._forecast):
                    timeval = bucket_start + forecast_pos * self.bucket_interval
                    dt = make_datetime(timeval)
                    dt_next[forecast_pos] = dt_get_weekday(dt)

                max_value = self.maxs[dt_pos]
                rng = max_value - self.mins[dt_pos]

                norm_dt = 1.0 - (max_value - dt_next) / rng
                X[:, 0:self._forecast, dt_pos] = norm_dt

            X = np.roll(X, -self._forecast, axis=1)

            Z_ = np.empty_like(Y_)
            for i, j, feature in self.enum_features(is_output=True):
                Z_[:,j] = _revert_scores(
                    feature,
                    Y_[:,j],
                    _data=None,
                    _min=self.mins[i],
                    _max=self.maxs[i],
                    _mean=self.means[i],
                    _std=self.stds[i],
                )

            for z in range(self._forecast):
                if bucket_start < to_ts:
                    timestamps.append(bucket_start)
                    predicted[bucket] = Z_[z][:]
                bucket_start += self.bucket_interval
                bucket += 1

        return TimeSeriesPrediction(
            self,
            timestamps=timestamps,
            observed=observed,
            predicted=predicted,
        )

    def get_scores(self, predicted, observed):
        """
        Compute scores and mean squared error
        """

        outputs = [feature for feature in self.features if feature.is_output]

        nb_output = len(self._y_indexes())

        # FIXME: change according to feature preprocessing. tip: for standardization, use
        # scoring function present in fingerprints_ok branch
        rng = self.maxs - self.mins

        X = observed
        Y = predicted

        X = 1.0 - (self.maxs[:nb_output] - X) / rng[:nb_output]
        Y = 1.0 - (self.maxs[:nb_output] - Y) / rng[:nb_output]

        diff = X - Y

        for i, feature in enumerate(outputs):
            ano_type = feature.anomaly_type

            if ano_type == 'low':
                diff[i] = -min(diff[i], 0)
            elif ano_type == 'high':
                diff[i] = max(diff[i], 0)
            else:
                diff[i] = abs(diff[i])

        mse = (diff ** 2).mean(axis=None)
        scores = np.clip(100 * diff, 0, 100)

        return scores, mse

    def detect_anomalies(self, prediction, hooks=[]):
        """
        Detect anomalies on observed data by comparing them to the values
        predicted by the model
        """

        stats = []
        anomaly_indices = []

        for i, ts in enumerate(prediction.timestamps):
            dt = ts_to_datetime(ts)
            date_str = datetime_to_str(dt)
            is_anomaly = False
            anomalies = {}

            predicted = prediction.predicted[i]
            observed = prediction.observed[i]

            scores, mse = self.get_scores(
                predicted,
                observed,
            )

            max_score = 0

            for j, score in enumerate(scores):
                feature = self.features[j]

                if not feature.is_output:
                    continue

                max_score = max(max_score, score)

                if score < self.max_threshold:
                    continue

                anomalies[feature.name] = {
                    'type': 'low' if observed[j] < predicted[j] else 'high',
                    'score': score,
                }

            if len(anomalies):
                is_anomaly = True
                anomaly_indices.append(i)

            anomaly = self._state.get('anomaly')

            if anomaly is None:
                if is_anomaly:
                    # This is a new anomaly

                    # TODO have a Model.logger to prefix all logs with model name
                    logging.warning("detected anomaly for model '%s' at %s (score = %.1f)",
                                    self.name, date_str, max_score)

                    self._state['anomaly'] = {
                        'start_ts': ts,
                        'max_score': max_score,
                    }

                    for hook in hooks:
                        logging.debug("notifying '%s' hook", hook.name)
                        data = prediction.format_bucket_data(i)

                        try:
                            hook.on_anomaly_start(
                                self.name,
                                dt=dt,
                                score=max_score,
                                predicted=data['predicted'],
                                observed=data['observed'],
                                anomalies=anomalies,
                            )
                        except Exception as exn:
                            # XXX: catch all the exception to avoid
                            # interruption
                            logging.exception(exn)
            else:
                if is_anomaly:
                    anomaly['max_score'] = max(anomaly['max_score'], max_score)
                    logging.warning(
                        "anomaly still in progress for model '%s' at %s (score = %.1f)",
                        self.name, date_str, max_score,
                    )
                elif score < self.min_threshold:
                    logging.info(
                        "anomaly ended for model '%s' at %s (score = %.1f)",
                        self.name, date_str, max_score,
                    )

                    for hook in hooks:
                        logging.debug("notifying '%s' hook", hook.name)
                        hook.on_anomaly_end(self.name, dt, max_score)

                    self._state['anomaly'] = None

            stats.append({
                'mse': mse,
                'score': max_score,
                'anomaly': is_anomaly,
                'anomalies': anomalies,
            })

        prediction.stats = stats
        prediction.anomaly_indices = anomaly_indices

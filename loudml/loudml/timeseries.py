"""
LoudML time-series module
"""

import json
import logging
import math
import os
import numpy as np

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
    Required,
    Range,
    Schema,
)

from . import (
    errors,
    schemas,
)
from .misc import (
    make_ts,
    ts_to_str,
    parse_timedelta,
)
from .model import (
    Model,
)

# global vars for easy reusability
# This UNIX process is handling a unique model
_keras_model, _graph = None, None
_mins, _maxs = None, None
_verbose = 0

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
            return self.model.defaults[feature_idx]
        return value

    def format_series(self):
        """
        Return prediction data as a time-series
        """

        observed = {}
        predicted = {}

        for i, feature in enumerate(self.model.features):
            default = self.model.defaults[i]

            observed[feature.name] = [self.apply_default(i, x) for x in self.observed[:,i]]
            predicted[feature.name] = [self.apply_default(i, x) for x in self.predicted[:,i]]

        result = {
            'timestamps': self.timestamps,
            'observed': observed,
            'predicted': predicted,
        }
        if self.stats is not None:
            result['stats'] = self.stats,
        return result

    def _format_bucket(self, i):
        """
        Format one bucket
        """

        features = self.model.features
        bucket = {
            'timestamp': self.timestamps[i],
            'observed': {
                feature.name: self.apply_default(j, self.observed[i][j])
                for j, feature in enumerate(features)
            },
            'predicted': {
                feature.name: self.apply_default(j, self.predicted[i][j])
                for j, feature in enumerate(features)
            },
        }
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
        Required('bucket_interval'): schemas.TimeDelta(min=0, min_included=False),
        Required('interval'): schemas.TimeDelta(min=0, min_included=False),
        Required('offset'): schemas.TimeDelta(min=0),
        Required('span'): All(int, Range(min=1)),
    })

    def __init__(self, settings, state=None):
        settings['type'] = self.TYPE
        super().__init__(settings, state)

        self.bucket_interval = parse_timedelta(settings.get('bucket_interval')).total_seconds()
        self.interval = parse_timedelta(settings.get('interval')).total_seconds()
        self.offset = parse_timedelta(settings.get('offset')).total_seconds()
        self.span = settings.get('span')
        self.sequential = None

        self.defaults = [
            None if feature.default is np.nan else feature.default
            for feature in self.features
        ]

    @property
    def type(self):
        return self.TYPE

    def _compute_nb_buckets(self, from_ts, to_ts):
        """
        Compute the number of bucket between `from_ts` and `to_ts`
        """
        return int((to_ts - from_ts) / self.bucket_interval) + 2

    def _train_on_dataset(
        self,
        dataset,
        train_size=0.67,
        batch_size=64,
        num_epochs=100,
        max_evals=None,
    ):
        global _mins, _maxs

        if max_evals is None:
            max_evals = self.settings.get('max_evals', 10)

        # Min-max preprocessing to bring data in interval (0,1)
        # FIXME: support other normalization techniques
        # Preprocess each column (axis=0)
        _mins = np.min(np.nan_to_num(dataset), axis=0)
        _maxs = np.max(np.nan_to_num(dataset), axis=0)
        rng = _maxs - _mins
        dataset = 1.0 - (_maxs - dataset) / rng
        nb_features = len(self.features)

        logging.info("Preprocessing. mins: %s maxs: %s ranges: %s",
                     _mins, _maxs, rng)

        (_, X_train, y_train), (_, X_test, y_test) = self.train_test_split(
            dataset,
            train_size=train_size,
        )

        def cross_val_model(params):
            global _keras_model, _graph
            _keras_model, _graph = None, None

            # Destroys the current TF graph and creates a new one.
            # Useful to avoid clutter from old models / layers.
            K.clear_session()

            # expected input data shape: (batch_size, timesteps, nb_features)
            _keras_model = Sequential()
            if params.depth == 1:
                _keras_model.add(LSTM(
                    params.l1,
                    input_shape=(None, nb_features),
                    return_sequences=False,
                ))
                _keras_model.add(Dense(nb_features, input_dim=params.l1))
            elif params.depth == 2:
                _keras_model.add(LSTM(
                    params.l1,
                    input_shape=(None, nb_features),
                    return_sequences=True,
                ))
                _keras_model.add(LSTM(params.l2, return_sequences=False))
                _keras_model.add(Dense(nb_features, input_dim=params.l2))

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
              'l1': 1+hp.randint('d1_l1', 100),
              'activation': hp.choice('d1_activation', ['tanh']),
              'loss_fct': hp.choice('d1_loss_fct', ['mean_squared_error']),
              'optimizer': hp.choice('d1_optimizer', ['adam']),
            },
            {
              'depth': 2,
              'l1': 1+hp.randint('d2_l1', 100),
              'l2': 1+hp.randint('d2_l2', 100),
              'activation': hp.choice('d2_activation', ['tanh']),
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

        for i in range(len(dataset) - self.span):
            j = i + self.span
            partX = dataset[i:j, :]
            partY = dataset[j, :]

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
        from_date=None,
        to_date=None,
        train_size=0.67,
        batch_size=64,
        num_epochs=100,
        max_evals=None,
    ):
        """
        Train model
        """
        global _keras_model, _graph, _mins, _maxs
        _keras_model, _graph = None, None
        _mins, _maxs = None, None

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

        # Fill dataset
        data = datasource.get_times_data(self, from_ts, to_ts)

        i = None
        for i, (_, val, _) in enumerate(data):
            dataset[i] = val

        if i is None:
            raise errors.NoData("no data found for time range {}-{}".format(
                from_str,
                to_str,
            ))

        logging.info("found %d time periods", i + 1)

        best_params, score, _, _ = self._train_on_dataset(
            dataset,
            train_size,
            batch_size,
            num_epochs,
            max_evals,
        )

        model_b64, weights_b64 = _serialize_keras_model(_keras_model)

        self.state = {
            'graph': model_b64,
            'weights': weights_b64, # H5PY data encoded in base64
            'loss_fct': best_params['loss_fct'],
            'optimizer': best_params['optimizer'],
            'best_params': best_params,
            'mins': _mins.tolist(),
            'maxs': _maxs.tolist(),
            'loss': score[0],
            'accuracy': score[1],

        }

        return {
            'loss': score[0],
            'accuracy': score[1],
        }

    def load(self):
        """
        Load current model
        """
        global _keras_model, _graph, _mins, _maxs

        if not self.is_trained:
            raise errors.ModelNotTrained()

        _keras_model, _graph = _load_keras_model(
            self.state['graph'],
            self.state['weights'],
            self.state['loss_fct'],
            self.state['optimizer'],
        )

        _mins = np.array(self.state['mins'])
        _maxs = np.array(self.state['maxs'])

    @property
    def is_trained(self):
        """
        Tells if model is trained
        """
        return self.state is not None and 'weights' in self.state

    def _format_dataset_predict(self, dataset):
        """
        Format dataset for time-series prediction

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

        Buckets with missing values are skipped.
        """
        data_x = []
        indexes = []

        for i in range(len(dataset) - self.span + 1):
            j = i + self.span
            partX = dataset[i:j, :]

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
        hist_from_ts = from_ts - self.span * self.bucket_interval
        hist_to_ts = to_ts
        hist_from_str = ts_to_str(hist_from_ts)
        hist_to_str = ts_to_str(hist_to_ts)

        # Prepare dataset
        nb_buckets = int((hist_to_ts - hist_from_ts) / self.bucket_interval)
        nb_features = len(self.features)
        dataset = np.zeros((nb_buckets, nb_features), dtype=float)
        X = []

        # Fill dataset
        logging.info("extracting data for range=[%s, %s]",
                     hist_from_str, hist_to_str)
        data = datasource.get_times_data(self, hist_from_ts, hist_to_ts)

        # Only a subset of history will be used for computing the prediction
        X_until = None # right bound for prediction
        i = None
        for i, (_, val, ts) in enumerate(data):
            dataset[i] = val

            if make_ts(ts) < to_ts - self.bucket_interval:
                X.append(ts)
                X_until = i + 1

        if i is None:
            raise errors.NoData("no data found for time range {}-{}".format(
                hist_from_str,
                hist_to_str,
            ))

        nb_buckets_found = i + 1
        if nb_buckets_found < nb_buckets:
            dataset = np.resize(dataset, (nb_buckets_found, nb_features))

        logging.info("found %d time periods", nb_buckets_found)

        rng = _maxs - _mins
        norm_dataset = 1.0 - (_maxs - dataset) / rng

        j_sel, X_test = self._format_dataset_predict(norm_dataset[:X_until])

        logging.info("generating prediction")
        Y_ = _keras_model.predict(X_test)

        # min/max inverse operation
        Z_ = _maxs - rng * (1.0 - Y_)

        # Build final result
        timestamps = X[self.span:]
        last_ts = make_ts(X[-1])
        timestamps.append(ts_to_str(last_ts + self.bucket_interval))

        shape = (predict_len, nb_features)
        observed = np.array([self.defaults] * predict_len)
        predicted = np.array([self.defaults] * predict_len)

        for i, feature in enumerate(self.features):
            observed[:,i] = dataset[self.span:][:,i]
            predicted[j_sel - self.span,i] = Z_[:][:,i]

        return TimeSeriesPrediction(
            self,
            timestamps=timestamps,
            observed=observed,
            predicted=predicted,
        )

    def detect_anomalies(self, prediction):
        """
        Detect anomalies on observed data by comparing them to the values
        predicted by the model
        """

        global _mins, _maxs

        nb_features = len(self.features)
        max_dist = np.linalg.norm(np.zeros(nb_features) - np.ones(nb_features))
        rng = _maxs - _mins

        stats = []
        anomaly_indices = []

        for i, ts in enumerate(prediction.timestamps):
            X = prediction.observed[i]
            Y = prediction.predicted[i]

            X = 1.0 - (_maxs - X) / rng
            Y = 1.0 - (_maxs - Y) / rng

            mse = ((X - Y) ** 2).mean(axis=None)
            dist = np.linalg.norm(X - Y)
            score = min((dist / max_dist) * 100, 100)

            if score >= self.threshold:
                # TODO have a Model.logger to prefix all logs with model name
                logging.warning("detected anomaly for %s (score = %.1f)",
                                ts, score)
                anomaly = True
                anomaly_indices.append(i)
            else:
                anomaly = False

            stats.append({
                'mse': mse,
                'dist': dist,
                'score': score,
                'anomaly': anomaly,
            })

        prediction.stats = stats
        prediction.anomaly_indices = anomaly_indices

"""
LoudML time-series module
"""

import logging
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

from . import (
    errors,
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

def _load_data(dataset, n_prev=1):
    data_x, data_y = [], []
    indexes = []

    for i in range(len(dataset)-n_prev):
        partX = dataset[i:(i+n_prev), :]
        partY = dataset[(i+n_prev), :]

        if not np.isnan(partX).any() and not np.isnan(partY).any():
            data_x.append(partX)
            data_y.append(partY)
            indexes.append(i)

    return np.array(indexes), np.array(data_x), np.array(data_y)

class TimesModel(Model):
    """
    Time-series model
    """

    def __init__(self, settings, state=None):
        super().__init__(settings, state)

        # TODO use voluptuous to check settings validity

        self.bucket_interval = int(parse_timedelta(settings.get('bucket_interval')).total_seconds())
        self.interval = int(parse_timedelta(settings.get('interval')).total_seconds())
        self.offset = int(parse_timedelta(settings.get('offset')).total_seconds())
        self.span = settings.get('span')
        self.sequential = None

    @property
    def type(self):
        return 'timeseries'

    def _compute_nb_buckets(self, from_ts, to_ts):
        """
        Compute the number of bucket between `from_ts` and `to_ts`
        """
        return int((to_ts - from_ts) / self.bucket_interval) + 1

    def _train_on_dataset(
        self,
        dataset,
        train_size=0.67,
        batch_size=64,
        num_epochs=100,
        max_evals=10,
    ):
        global _mins, _maxs

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
        Format dataset for time-series prediction

        It is assumed that a value for a given bucket can be predicted
        according the preceding ones. The number of preceding buckets used
        for prediction is given by `self.span`.

        input:
        [v0, v1, v2, v3, ,v4 ..., vn]

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
        max_evals=10,
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
        i = 0
        for i, (_, val, _) in enumerate(data):
            dataset[i] = val

        if i == 0:
            raise errors.NoData("no data found for time range {}-{}".format(
                from_str,
                to_str,
            ))

        logging.info("found %d time periods", i + 1)

        best_params, _, _, _ = self._train_on_dataset(
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
        }

    @property
    def is_trained(self):
        """
        Tells if model is trained
        """
        return self.state and 'weights' in self.state

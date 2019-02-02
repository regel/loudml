"""Loud ML VAE time series model

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.

# Reference:
- [Unsupervised Anomaly Detection via Variational Auto-Encoder](
    https://arxiv.org/abs/1802.03903)
"""


import copy
import datetime
import json
import logging
import os
import sys
import random
import time
import numpy as np
import itertools
from scipy.stats import norm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras.callbacks import EarlyStopping
from tensorflow.contrib.keras.api.keras.layers import Lambda, Input, Dense
from tensorflow.contrib.keras.api.keras.models import Model as _Model
from tensorflow.contrib.keras.api.keras.losses import mean_squared_error
from tensorflow.contrib.keras.api.keras import regularizers
from tensorflow.contrib.keras.api.keras.utils import plot_model

import h5py # Read training_config.optimizer_config

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
    list_from_np,
    make_datetime,
    make_ts,
    nan_to_none,
    parse_timedelta,
    ts_to_str,
    ts_to_datetime,
)
from .model import (
    Model,
    DateRange,
)

DEFAULT_SEASONALITY = {
    'daytime': False,
    'weekday': False,
}

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

_verbose = 0

_hp_span_min = 10
_hp_span_max = 100

# Constants derived from https://arxiv.org/abs/1802.03903
g_mcmc_count = 10
g_mc_count = 1000
g_mc_batch_size = 256
g_lambda = 0.01


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def add_loss(model, W):
    inputs = model.inputs[0]
    abnormal = model.inputs[1]
    # abnormal = K.print_tensor(abnormal, message='abnormal = ')
    outputs = model.outputs[0]
    z_mean = model.get_layer('z_mean').output
    z_log_var = model.get_layer('z_log_var').output

    beta = K.sum(1.0 - abnormal, axis=-1, keepdims=True) / W
    # beta = K.print_tensor(beta, message='beta = ')
    reconstruction_loss = mean_squared_error(inputs, outputs)
    reconstruction_loss *= W
    kl_loss = 1 + z_log_var - beta * K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    model.add_loss(vae_loss)

def _get_encoder(_keras_model):
    # instantiate encoder model
    main_input = _keras_model.inputs[0]
    aux_input = _keras_model.inputs[1]
    z_mean = _keras_model.get_layer('z_mean').output
    z_log_var = _keras_model.get_layer('z_log_var').output
    z = _keras_model.get_layer('z').output
    model = _Model([main_input, aux_input], [z_mean, z_log_var, z], name='encoder')
    return model

def _get_decoder(_keras_model):
    # instantiate decoder model
    _, latent_dim = _keras_model.get_layer('z').output.get_shape()
    latent_dim = int(latent_dim)
    latent_inputs = Input(shape=(int(latent_dim),), name='z_sampling')
    x = _keras_model.get_layer('dense_1')(latent_inputs)
    output = _keras_model.get_layer('dense_2')(x)
    new_model = _Model(latent_inputs, output, name='decoder')
    return new_model

def _get_index(d, from_date, step):
    return int((make_ts(d) - make_ts(from_date)) / step)

def _format_windows(from_date, to_date, step, windows):
    size = _get_index(to_date, from_date, step)
    abnormal = np.full((size,), False, dtype=bool)
    for _from, _to in windows:
        x = _get_index(_from, from_date, step)
        y = _get_index(_to, from_date, step)
        abnormal[min(size,max(0,x)):max(0,min(y,size))] = True

    return abnormal

def _get_scores(y, _mean, _std):
    y = (y - _mean) / _std
    return y

def _revert_scores(y, _mean, _std):
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

    fd, path = tempfile.mkstemp()
    try:
        keras_model.save(path)
        with os.fdopen(fd, 'rb') as tmp:
            model_b64 = base64.b64encode(tmp.read())
    finally:
        os.remove(path)

    return model_b64.decode('utf-8')

def _load_keras_model(model_b64):
    import tempfile
    import base64

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(base64.b64decode(model_b64.encode('utf-8')))
            tmp.close()
    finally:
        keras_model = load_model(path, compile=False)
        opened_new_file = not isinstance(path, h5py.File)
        if opened_new_file:
            f = h5py.File(path, mode='r')
        else:
            f = filepath
        training_config = f.attrs.get('training_config')
        optimizer_cls = None
        if training_config is None:
            optimizer_cls = tf.keras.optimizers.Adam()
        else:
            training_config = json.loads(training_config.decode('utf-8'))
            optimizer_config = training_config['optimizer_config']
            optimizer_cls = tf.keras.optimizers.deserialize(optimizer_config)

        if opened_new_file:
            f.close()
        os.remove(path)
        _, W = keras_model.inputs[0].get_shape()
        add_loss(keras_model, int(W))
        keras_model.compile(
            optimizer=optimizer_cls,
        )

    return keras_model

class TimeSeriesPrediction:
    """
    Time-series prediction
    """

    def __init__(self, model, timestamps, observed, predicted, upper=None, lower=None):
        self.model = model
        self.timestamps = timestamps
        self.observed = observed
        self.predicted = predicted
        self.upper = upper
        self.lower = lower
        self.anomaly_indices = None
        self.stats = None
        self.constraint = None
        self.scores = None
        self.mses = None
        self.mse = None

    def get_anomalies(self):
        """
        Return anomalies
        """

        if self.anomaly_indices is None:
            raise errors.NotFound("anomaly detection has not been performed yet")
        return [self._format_bucket(i) for i in self.anomaly_indices]

    def format_series(self):
        """
        Return prediction data as a time-series
        """

        observed = {}
        predicted = {}
        
        feature = self.model.features[0]
        observed[feature.name] = list_from_np(self.observed)
        predicted[feature.name] = list_from_np(self.predicted)
        if self.lower is not None:
            predicted['lower_{}'.format(feature.name)] = list_from_np(self.lower)
        if self.upper is not None:
            predicted['upper_{}'.format(feature.name)] = list_from_np(self.upper)

        result = {
            'timestamps': self.timestamps,
            'observed': observed,
            'predicted': predicted,
        }
        if self.stats is not None:
            result['stats'] = self.stats
        if self.constraint is not None:
            result['constraint'] = self.constraint
        return result

    def get_field_names(self):
        features = self.model.features
        names = []
        for feature in features:
            names.append(feature.name)
            names.append("lower_{}".format(feature.name))
            names.append("upper_{}".format(feature.name))

        return names

    def format_bucket_data(self, i):
        """
        Format observation and prediction for one bucket
        """
        feature = self.model.features[0]
        predicted = {
            feature.name: nan_to_none(self.predicted[i])
        }
        if self.lower is not None:
            predicted.update({
                'lower_{}'.format(feature.name): nan_to_none(self.lower[i])
            })
        if self.upper is not None:
            predicted.update({
                'upper_{}'.format(feature.name): nan_to_none(self.upper[i])
            })
        return {
            'observed': {
                feature.name: nan_to_none(self.observed[i])
            },
            'predicted': predicted
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

    def stat(self):
        self.scores, self.mses = self.model.compute_scores(
            self.observed,
            self.predicted,
            self.lower,
            self.upper,
            )
        self.mse = np.nanmean(self.mses, axis=None)

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

        x = np.linspace(1, len(self.observed), len(self.observed))
        plt.rcParams["figure.figsize"] = (17, 9)
        plt.plot(x, self.observed, "--")
        plt.fill_between(x, self.lower, self.upper, alpha=0.2)
        plt.plot(x, self.predicted, ":", color='black')
        plt.show()


def generator(x, missing, batch_size, model):
    batch_x = np.zeros((batch_size, x.shape[1]))
    batch_missing = np.zeros((batch_size, x.shape[1]))
    while True:
        abnormal = np.random.binomial(1, g_lambda, x.shape[1])
        for i in range(batch_size):
            index = random.randint(0,len(x)-1)
            batch_x[i] = x[index]
            batch_missing[i] = np.maximum(
                abnormal,
                missing[index],
            )

        for _ in range(g_mcmc_count):
            x_decoded, _ = model.predict([batch_x, batch_missing], batch_size=g_mc_batch_size)
            batch_x[batch_missing > 0] = x_decoded[batch_missing > 0]
       
        yield ([batch_x, batch_missing], None)

class DonutModel(Model):
    """
    Time-series VAE model, "Donut"
    """
    TYPE = 'donut'

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
        Optional('grace_period', default=0): schemas.TimeDelta(min=0, min_included=True),
        'default_datasink': schemas.key,
    })

    def __init__(self, settings, state=None):
        global _hp_span_min, _hp_span_max
        super().__init__(settings, state)

        settings = self.validate(settings)
        self.bucket_interval = parse_timedelta(settings.get('bucket_interval')).total_seconds()
        self.interval = parse_timedelta(settings.get('interval')).total_seconds()
        self.offset = parse_timedelta(settings.get('offset')).total_seconds()

        self.span = settings.get('span')

        self.means = None
        self.stds = None
        self.scores = None
        self._keras_model = None
        self._encoder_model = None
        self._decoder_model = None

        if self.span is None or self.span == "auto":
            self.min_span = settings.get('min_span') or _hp_span_min
            self.max_span = settings.get('max_span') or _hp_span_max
        else:
            self.min_span = self.span
            self.max_span = self.span

        self.grace_period = parse_timedelta(settings['grace_period']).total_seconds()

        self.current_eval = None
        if len(self.features) > 1:
            raise errors.LoudMLException("This model type supports one unique feature")

    def enum_features(self, is_input=None, is_output=None):
        j = 0
        for i, feature in enumerate(self.features):
            if feature.is_input == is_input or feature.is_output == is_output:
                yield i, j, feature
                j += 1

    @property
    def type(self):
        return self.TYPE

    @property
    def W(self):
        return self.span

    def get_hp_span(self, label):
        if (self.max_span - self.min_span) <= 0:
            space = self.span
        else:
            space = self.min_span + hp.randint(label, (self.max_span - self.min_span))
        return space

    def set_run_params(self, params=None):
        """
        Set running parameters to make them persistent
        """
        if params is None:
            self._settings.pop('run', None)
        else:
            self._settings['run'] = params

    def set_run_state(self, params=None):
        """
        Set running forecast parameters to make them persistent
        """
        if params is None:
            self._state.pop('run', None)
        else:
            self._state['run'] = params

    def get_run_state(self):
        return self._state.get('run') or {}

    def compute_nb_buckets(self, from_ts, to_ts):
        """
        Compute the number of bucket between `from_ts` and `to_ts`
        """
        return int((to_ts - from_ts) / self.bucket_interval) + 2

    def apply_defaults(self, x):
        """
        Apply default feature value to np array
        """
        feature = self.features[0]
        if feature.default == "previous":
            previous = None
            for j, value in enumerate(x):
                if np.isnan(value):
                    x[j] = previous
                else:
                    previous = x[j]
        elif not np.isnan(feature.default):
            x[np.isnan(x)] = feature.default

    def scale_dataset(
        self,
        dataset,
    ):
        """
        Scale dataset values
        """

        out = _get_scores(
            dataset,
            _mean=self.means[0],
            _std=self.stds[0],
        )

        return out

    def unscale_dataset(
        self,
        dataset,
    ):
        """
        Revert scaling dataset values
        """

        out = _revert_scores(
            dataset,
            _mean=self.means[0],
            _std=self.stds[0],
        )

        return out

    def stat_dataset(self, dataset):
        """
        Compute dataset sets and keep them as reference
        """
        self.means = np.array([np.nanmean(dataset, axis=0)])
        self.stds = np.array([np.nanstd(dataset, axis=0)])
        self.stds[self.stds == 0] = 1.0

    def set_auto_threshold(self):
        """
        Compute best threshold values automatically
        """
        # 68–95–99.7 three-sigma rule
        self.min_threshold = 68
        self.max_threshold = 99.7

    def _set_xpu_config(self, num_cpus, num_gpus):
        config = tf.ConfigProto(
            allow_soft_placement=True,
            device_count = {'CPU' : num_cpus, 'GPU' : num_gpus},
        )
        config.gpu_options.allow_growth = True
#        config.log_device_placement = True
#        config.intra_op_parallelism_threads=num_cores
#        config.inter_op_parallelism_threads=num_cores
                
        sess = tf.Session(config=config)
        K.set_session(sess)

    def _train_on_dataset(
        self,
        dataset,
        train_size=0.67,
        batch_size=64,
        num_epochs=100,
        num_cpus=1,
        num_gpus=0,
        max_evals=None,
        progress_cb=None,
        abnormal=None,
    ):
        if max_evals is None:
            max_evals = self.settings.get('max_evals', 21) # latent_dim*intermediate_dim

        self.current_eval = 0

        self.stat_dataset(dataset)
        dataset = self.scale_dataset(dataset)

        def cross_val_model(params):
            keras_model = None
            # Destroys the current TF graph and creates a new one.
            # Useful to avoid clutter from old models / layers.
            K.clear_session()
            self._set_xpu_config(num_cpus, num_gpus)

            self.span = W = params.span
            (X_miss, X_train), (X_miss_val, X_test) = self.train_test_split(
                dataset,
                train_size=train_size,
                abnormal=abnormal,
            )
            if len(X_train) == 0:
                raise errors.NoData("insufficient training data")
            if len(X_test) == 0:
                raise errors.NoData("insufficient validation data")

            # expected input data shape: (batch_size, timesteps,)
            # network parameters
            input_shape = (W, )
            intermediate_dim = params.intermediate_dim
            latent_dim = params.latent_dim
            
            # VAE model = encoder + decoder
            # build encoder model
            main_input = Input(shape=input_shape)
            aux_input = Input(shape=input_shape) # bool vector to flag missing data points
            aux_output = Lambda(lambda x: x)(aux_input)
            x = Dense(intermediate_dim,
                      kernel_regularizer=regularizers.l2(0.01),
                      activation='relu')(main_input)
            z_mean = Dense(latent_dim, name='z_mean')(x)
            z_log_var = Dense(latent_dim, name='z_log_var')(x)
            
            # use reparameterization trick to push the sampling out as input
            # note that "output_shape" isn't necessary with the TensorFlow backend
            z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
            
            # build decoder model
            x = Dense(intermediate_dim,
                      kernel_regularizer=regularizers.l2(0.01),
                      activation='relu', name='dense_1')(z)
            main_output = Dense(W, activation='linear', name='dense_2')(x)
             
            # instantiate Donut model
            keras_model = _Model([main_input, aux_input], [main_output, aux_output], name='donut')
            add_loss(keras_model, W)
            optimizer_cls = None
            if params.optimizer == 'adam':
                optimizer_cls = tf.keras.optimizers.Adam()

            keras_model.compile(
                optimizer=optimizer_cls,
            )

            _stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                verbose=_verbose,
                mode='auto',
            )
            keras_model.fit_generator(
                generator(X_train, X_miss, batch_size, keras_model),
                epochs=num_epochs,
                steps_per_epoch=len(X_train) / batch_size,
                verbose=_verbose,
                validation_data=([X_test, X_miss_val], None),
                callbacks=[_stop],
                workers=0, # https://github.com/keras-team/keras/issues/5511
            )

            # How well did it do?
            score = keras_model.evaluate(
                [X_test, X_miss_val],
                batch_size=batch_size,
                verbose=_verbose,
            )

            self.current_eval += 1
            if progress_cb is not None:
                progress_cb(self.current_eval, max_evals)

            return score, keras_model

        hyperparameters = HyperParameters()

        # Parameter search space
        def objective(args):
            hyperparameters.assign(args)

            try:
                score, _ = cross_val_model(hyperparameters)
                return {'loss': score, 'status': STATUS_OK}
            except Exception as exn:
                logging.warning("iteration failed: %s", exn)
                return {'loss': None, 'status': STATUS_FAIL}

        space = hp.choice('case', [
            {
              'span': self.get_hp_span('span'),
              'latent_dim': hp.choice('latent_dim', [3, 5, 8]),
              'intermediate_dim': hp.choice('i1', [21, 34, 55, 89, 144, 233, 377]),
              'optimizer': hp.choice('optimizer', ['adam']),
            }
        ])

        # The Trials object will store details of each iteration
        trials = Trials()

        # Run the hyperparameter search using the tpe algorithm
        try:
            best = fmin(
                objective,
                space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
            )
        except ValueError:
            raise errors.NoData("training failed, try to increase the time range")

        # Get the values of the optimal parameters
        best_params = space_eval(space, best)
        score, self._keras_model = cross_val_model(
            HyperParameters(best_params)
        )
        self.span = best_params['span']
        return (best_params, score)

    def _train_ckpt_on_dataset(
        self,
        dataset,
        train_size=0.67,
        batch_size=64,
        num_epochs=100,
        progress_cb=None,
        abnormal=None,
    ):
        self.current_eval = 0
        self.stat_dataset(dataset)

        dataset = self.scale_dataset(dataset)

        (X_miss, X_train), (X_miss_val, X_test) = self.train_test_split(
            dataset,
            train_size=train_size,
        )

        _stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=_verbose,
            mode='auto',
        )
        self._keras_model.fit(
            [X_train, X_miss],
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=_verbose,
            validation_data=([X_test, X_miss_val], None),
            callbacks=[_stop],
        )

        # How well did it do?
        score = self._keras_model.evaluate(
            [X_test, X_miss_val],
            batch_size=batch_size,
            verbose=_verbose,
        )
        return score

    def compute_bucket_scores(self, y_true, y_pred, y_low, y_high):
        """
        Compute scores and mean squared error
        """
        feature = self.features[0]

        diff = y_true - y_pred
        ano_type = feature.anomaly_type
        mu = (y_low + y_high) / 2.0
        std = (y_high - mu) / 3.0
        score = 2 * norm.cdf(abs(y_true - mu), loc=0, scale=std) - 1
        # Required to handle the 'low' condition
        if diff < 0:
            score *= -1

        if ano_type == 'low':
            score = -min(score, 0)
        elif ano_type == 'high':
            score = max(score, 0)
        else:
            score = abs(score)

        score = 100 * max(0, min(1, score))

        mse = np.nanmean((diff ** 2), axis=None)
        return score, mse

    def compute_scores(self, observed, predicted, low, high):
        """
        Compute timeseries scores and MSE
        """

        nb_buckets = len(observed)
        scores = np.empty((nb_buckets,), dtype=float)
        mses = np.empty((nb_buckets), dtype=float)

        for i in range(nb_buckets):
            scores[i], mses[i] = self.compute_bucket_scores(
                observed[i],
                predicted[i],
                low[i],
                high[i],
            )

        return scores, mses

    def _format_dataset(self, x, accept_missing=True, abnormal=None):
        """
        Format dataset for time-series training & inference

        input:
        [v0, v1, v2, v3, v4 ..., vn]

        len: W

        output:
        missing = [0, 0, 1..., 0]
        X = [
            [v0, v1, v2], # span = W
            [v1, v2, v3],
            [v2, v3, v4],
            ...
            [..., .., vn],
        ]

        Buckets with missing values are flagged in the missing array.
        """
        missing = []
        data_x = []
        for i in range(len(x) - self.W + 1):
            j = i + self.W
            if (accept_missing == True) or (not np.isnan(x[i:j]).any()):
                # arxiv.org/abs/1802.03903
                # set user defined abnormal data points to zero
                if abnormal is None:
                    is_nan = np.isnan(x[i:j])
                else:
                    is_nan = np.logical_or(
                        np.isnan(x[i:j]),
                        abnormal[i:j],
                        )

                missing.append(is_nan)
                _x = np.copy(x[i:j])
                # set missing points to zero
                _x[is_nan == True] = 0.0
                data_x.append(_x)

        return np.array(missing), np.array(data_x)

    def train_test_split(self, dataset, abnormal=None, train_size=0.67):
        """
        Splits data to training and testing parts
        """
        ntrn = round(len(dataset) * train_size)
        X_train_missing, X_train = self._format_dataset(dataset[0:ntrn], abnormal=abnormal)
        X_test_missing, X_test = self._format_dataset(dataset[ntrn:])
        return (X_train_missing, X_train), (X_test_missing, X_test)

    def train(
        self,
        datasource,
        from_date,
        to_date="now",
        train_size=0.67,
        batch_size=256,
        num_epochs=100,
        num_cpus=1,
        num_gpus=0,
        max_evals=None,
        progress_cb=None,
        license=None,
        incremental=False,
        windows=[],
    ):
        """
        Train model
        """

        self.means, self.stds = None, None
        self.scores = None

        period =  self.build_date_range(from_date, to_date)
        logging.info(
            "train(%s) range=%s train_size=%f batch_size=%d epochs=%d)",
            self.name,
            period,
            train_size,
            batch_size,
            num_epochs,
        )

        # Prepare dataset
        nb_buckets = self.compute_nb_buckets(period.from_ts, period.to_ts)
        dataset = np.full((nb_buckets,), np.nan, dtype=float)
        abnormal = _format_windows(
            period.from_ts,
            period.to_ts,
            self.bucket_interval,
            windows,
        )

        # Fill dataset
        data = datasource.get_times_data(self, period.from_ts, period.to_ts)
        # FIXME: query abnormal points flagged

        i = None
        for i, (_, val, timeval) in enumerate(data):
            dataset[i] = val

        if i is None:
            raise errors.NoData("no data found for time range {}".format(period))

        self.apply_defaults(dataset)

        nb_buckets_found = i + 1
        if nb_buckets_found < nb_buckets:
            dataset = np.resize(dataset, (nb_buckets_found,))

        logging.info("found %d time periods", nb_buckets_found)

        if incremental == True:
            best_params = self._state.get('best_params', dict())
            # Destroys the current TF graph and creates a new one.
            # Useful to avoid clutter from old models / layers.
            self.load(num_cpus, num_gpus)
            score = self._train_ckpt_on_dataset(
                dataset,
                train_size,
                batch_size,
                num_epochs,
                progress_cb=progress_cb,
                abnormal=abnormal,
            )
        else:
            best_params, score = self._train_on_dataset(
                dataset,
                train_size,
                batch_size,
                num_epochs,
                num_cpus,
                num_gpus,
                max_evals,
                progress_cb=progress_cb,
                abnormal=abnormal,
            )
        self.current_eval = None

        for key, val in best_params.items():
            if not isinstance(val, str) and \
               not isinstance(val, int) and \
               not isinstance(val, float):
                best_params[key] = np.asscalar(val)

        model_b64 = _serialize_keras_model(self._keras_model)

        self._state = {
            'h5py': model_b64,
            'best_params': best_params,
            'means': self.means.tolist(),
            'stds': self.stds.tolist(),
            'loss': score,
        }
        self.unload()
        #prediction = self.predict(
        #    datasource,
        #    from_date,
        #    to_date,
        #    num_cpus=num_cpus,
        #    num_gpus=num_gpus,
        #)
        #prediction.stat()

        return {
            'loss': score,
        }

    def unload(self):
        """
        Unload current model
        """
        self._keras_model = None
        self._encoder_model = None
        self._decoder_model = None
        K.clear_session()

    def load(self, num_cpus, num_gpus):
        """
        Load current model
        """
        if not self.is_trained:
            raise errors.ModelNotTrained()
        if self._keras_model:
            # Already loaded
            return

        K.clear_session()
        self._set_xpu_config(num_cpus, num_gpus)

        if self._state.get('h5py', None) is not None:
            self._keras_model = _load_keras_model(self._state.get('h5py'))
            # instantiate encoder model
            self._encoder_model = _get_encoder(self._keras_model)
            # instantiate decoder model
            self._decoder_model = _get_decoder(self._keras_model)
        else:
            raise errors.ModelNotTrained()

        if 'means' in self._state:
            self.means = np.array(self._state['means'])
        if 'stds' in self._state:
            self.stds = np.array(self._state['stds'])
        if 'scores' in self._state:
            self.scores = np.array(self._state['scores'])
        if self.min_threshold == 0 and self.max_threshold == 0:
            self.set_auto_threshold()
            logging.info(
                "setting threshold range min=%f max=%f",
                self.min_threshold,
                self.max_threshold,
            )


    @property
    def is_trained(self):
        """
        Tells if model is trained
        """
        return self._state is not None and ('weights' in self._state or 'h5py' in self._state)

    @property
    def _span(self):
        if self._state and 'span' in self._state['best_params']:
            return self._state['best_params']['span']
        else:
            return self.span

    @property
    def _window(self):
        return self._span

    def predict(
        self,
        datasource,
        from_date,
        to_date,
        license=None,
        num_cpus=1,
        num_gpus=0,
    ):
        global g_mcmc_count
        global g_mc_count
        global g_mc_batch_size

        period = self.build_date_range(from_date, to_date)

        # This is the number of buckets that the function MUST return
        predict_len = int((period.to_ts - period.from_ts) / self.bucket_interval)

        logging.info("predict(%s) range=%s", self.name, period)

        self.load(num_cpus, num_gpus)

        # Build history time range
        # Extra data are required to predict first buckets
        _window = self._window - 1

        hist = DateRange(
            period.from_ts - _window * self.bucket_interval,
            period.to_ts,
        )

        # Prepare dataset
        nb_buckets = int((hist.to_ts - hist.from_ts) / self.bucket_interval)
        dataset = np.full((nb_buckets,), np.nan, dtype=float)
        X = []

        # Fill dataset
        logging.info("extracting data for range=%s", hist)
        data = datasource.get_times_data(self, hist.from_ts, hist.to_ts)

        # Only a subset of history will be used for computing the prediction
        X_until = None # right bound for prediction
        i = None

        for i, (_, val, timeval) in enumerate(data):
            dataset[i] = val

            dt = make_datetime(timeval)
            ts = dt.timestamp()
            if ts < period.to_ts:
                X.append(make_ts(timeval))
                X_until = i + 1

        if i is None:
            raise errors.NoData("no data found for time range {}".format(hist))

        self.apply_defaults(dataset)

        nb_buckets_found = i + 1
        if nb_buckets_found < nb_buckets:
            dataset = np.resize(dataset, (nb_buckets_found,))

        logging.info("found %d time periods", nb_buckets_found)

        real = np.copy(dataset)

        norm_dataset = self.scale_dataset(dataset)
        missing, X_test = self._format_dataset(norm_dataset[:X_until])
        if len(X_test) == 0:
            raise errors.LoudMLException("not enough data for prediction")

        # force last col to missing
        missing[:, -1] = True

        logging.info("generating prediction")
        x_ = X_test.copy()
        # MCMC
        for _ in range(g_mcmc_count):
            z_mean, _, _ = self._encoder_model.predict([x_, missing], batch_size=g_mc_batch_size)
            x_decoded = self._decoder_model.predict(z_mean, batch_size=g_mc_batch_size)
            x_[missing == True] = x_decoded[missing == True]

        y = np.full((predict_len,), np.nan, dtype=float)
        y_low = np.full((predict_len,), np.nan, dtype=float)
        y_high = np.full((predict_len,), np.nan, dtype=float)
        no_missing_point = np.full((g_mc_count, self.W), False, dtype=bool)
        for j, x in enumerate(x_):
            y[j] = x[-1]
            # MC integration
            _, _, Z = self._encoder_model.predict(
                [np.tile(x, [g_mc_count, 1]), no_missing_point],
                batch_size=g_mc_batch_size,
            )
            x_decoded = self._decoder_model.predict(Z, batch_size=g_mc_batch_size)
            std = np.std(x_decoded[:,-1])
            y_low[j] = x[-1] - 3 * std
            y_high[j] = x[-1] + 3 * std

        y = self.unscale_dataset(y)
        y_low = self.unscale_dataset(y_low)
        y_high = self.unscale_dataset(y_high)

        # Build final result
        timestamps = X[_window:]

        shape = (predict_len, len(self.features))
        observed = np.full(shape, np.nan, dtype=float)
        observed = real[_window:]
        self.apply_defaults(observed)
        self.apply_defaults(y)

        return TimeSeriesPrediction(
            self,
            timestamps=timestamps,
            observed=observed,
            predicted=y,
            lower=y_low,
            upper=y_high,
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
        percent_interval=0.68,
        percent_noise=0,
        license=None,
        num_cpus=1,
        num_gpus=0,
    ):
        global g_mcmc_count
        global g_mc_count
        global g_mc_batch_size

        period = self.build_date_range(from_date, to_date)

        # This is the number of buckets that the function MUST return
        forecast_len = int((period.to_ts - period.from_ts) / self.bucket_interval)

        logging.info("forecast(%s) range=%s", self.name, period)

        self.load(num_cpus, num_gpus)

        # Build history time range
        # Extra data are required to predict first buckets
        _window = self._window - 1

        hist = DateRange(
            period.from_ts - _window * self.bucket_interval,
            period.to_ts,
        )

        # Prepare dataset
        nb_buckets = int((hist.to_ts - hist.from_ts) / self.bucket_interval)
        dataset = np.full((nb_buckets,), np.nan, dtype=float)
        X = []

        # Fill dataset
        logging.info("extracting data for range=%s", hist)
        data = datasource.get_times_data(self, hist.from_ts, hist.to_ts)

        # Only a subset of history will be used for computing the prediction
        X_until = None # right bound for prediction
        i = None

        for i, (_, val, timeval) in enumerate(data):
            dataset[i] = val

            dt = make_datetime(timeval)
            ts = dt.timestamp()
            if ts < period.to_ts:
                X.append(make_ts(timeval))
                X_until = i + 1

        if i is None:
            raise errors.NoData("no data found for time range {}".format(hist))

        self.apply_defaults(dataset)

        nb_buckets_found = i + 1
        if nb_buckets_found < nb_buckets:
            dataset = np.resize(dataset, (nb_buckets_found,))

        logging.info("found %d time periods", nb_buckets_found)

        real = np.copy(dataset)

        norm_dataset = self.scale_dataset(dataset)
        _, X_test = self._format_dataset(norm_dataset[:X_until])
        if len(X_test) == 0:
            raise errors.LoudMLException("not enough data for prediction")

        logging.info("generating prediction")
        x_ = X_test.copy()

        p = norm().ppf(1-(1-percent_interval)/2)
        missing = np.full((self._window,), False, dtype=bool)
        # force last col to missing
        missing[-1] = True
        y = np.full((forecast_len,), np.nan, dtype=float)
        y_low = np.full((forecast_len,), np.nan, dtype=float)
        y_high = np.full((forecast_len,), np.nan, dtype=float)
        x = x_[0]
        noise = percent_noise * float(self.bucket_interval) / (24*3600)
        for j, _ in enumerate(x_):
            # MCMC
            for _ in range(g_mcmc_count):
                z_mean, _, _ = self._encoder_model.predict(
                    [np.array([x]), np.array([missing])],
                    batch_size=g_mc_batch_size,
                )
                x_decoded = self._decoder_model.predict(z_mean, batch_size=g_mc_batch_size)
                x[missing == True] = x_decoded[0][missing == True]

            # uncertainty is modeled using a random uniform noise distribution
            # that increases over time
            expand = np.random.uniform(-noise * j, noise * j, len(x))
            x *= 1 + expand
            # MC integration
            _, _, Z = self._encoder_model.predict(
                [np.tile(x, [g_mc_count, 1]), np.tile(missing, [g_mc_count, 1])],
                batch_size=g_mc_batch_size,
            )
            x_decoded = self._decoder_model.predict(Z, batch_size=g_mc_batch_size)
            std = np.std(x_decoded[:,-1])
            y_low[j] = x[-1] - p * std
            y_high[j] = x[-1] + p * std
            y[j] = x[-1]
            x = np.roll(x, -1)
            # set missing point to zero
            x[-1] = 0

        y = self.unscale_dataset(y)
        y_low = self.unscale_dataset(y_low)
        y_high = self.unscale_dataset(y_high)

        # Build final result
        timestamps = X[_window:]

        shape = (forecast_len, len(self.features))
        observed = np.full(shape, np.nan, dtype=float)
        observed = real[_window:]
        self.apply_defaults(observed)
        self.apply_defaults(y)

        return TimeSeriesPrediction(
            self,
            timestamps=timestamps,
            observed=observed,
            predicted=y,
            lower=y_low,
            upper=y_high,
        )

    def detect_anomalies(self, prediction, hooks=[]):
        """
        Detect anomalies on observed data by comparing them to the values
        predicted by the model
        """

        prediction.stat()
        stats = []
        anomaly_indices = []

        for i, ts in enumerate(prediction.timestamps):
            last_anomaly_ts = self._state.get('last_anomaly_ts', 0)

            in_grace_period = (ts - last_anomaly_ts) < self.grace_period

            dt = ts_to_datetime(ts)
            date_str = datetime_to_str(dt)
            is_anomaly = False
            anomalies = {}

            predicted = prediction.predicted[i]
            observed = prediction.observed[i]

            score = prediction.scores[i]
            mse = prediction.mses[i]

            max_score = 0
            feature = self.features[0]
            max_score = max(max_score, score)

            if (not in_grace_period) and score >= self.max_threshold:
                anomalies[feature.name] = {
                    'type': 'low' if observed < predicted else 'high',
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
                    logging.warning(
                        "detected anomaly for model '%s' at %s (score = %.1f)",
                        self.name, date_str, max_score,
                    )

                    self._state['anomaly'] = {
                        'start_ts': ts,
                        'max_score': max_score,
                    }

                    for hook in hooks:
                        logging.debug("notifying '%s' hook", hook.name)
                        data = prediction.format_bucket_data(i)

                        try:
                            hook.on_anomaly_start(
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
                        hook.on_anomaly_end(dt, max_score)

                    self._state['anomaly'] = None
                    self._state['last_anomaly_ts'] = ts

            stats.append({
                'mse': nan_to_none(mse),
                'score': max_score,
                'anomaly': is_anomaly,
                'anomalies': anomalies,
            })

        prediction.stats = stats
        prediction.anomaly_indices = anomaly_indices


    def predict2(
        self,
        datasource,
        from_date,
        to_date,
        mse_rtol,
        _state={},
        license=None,
        num_cpus=1,
        num_gpus=0,
    ):
        return self.predict(
            datasource,
            from_date,
            to_date,
            license=license,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        )

    def plot_results(
        self,
        datasource,
        from_date,
        to_date,
        license=None,
        num_cpus=1,
        num_gpus=0,
        x_dim=-1,
        y_dim=-1,
        output=None,
    ):
        """
    
        # Arguments:
            models (tuple): encoder and decoder models
            data (tuple): test data and label
            model_name (string): which model is using this function
        """
        global g_mc_batch_size
        # Agg = Anti-grain geometry engine
        # running inside a Docker image. No Xwindow
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d

        period = self.build_date_range(from_date, to_date)

        logging.info("plot_results(%s) range=%s", self.name, period)

        self.load(num_cpus, num_gpus)
        _, latent_dim = self._encoder_model.outputs[0].get_shape()

        # Build history time range
        # Extra data are required to predict first buckets
        _window = self._window - 1

        hist = DateRange(
            period.from_ts - _window * self.bucket_interval,
            period.to_ts,
        )

        # Prepare dataset
        nb_buckets = int((hist.to_ts - hist.from_ts) / self.bucket_interval)
        dataset = np.full((nb_buckets,), np.nan, dtype=float)

        # Fill dataset
        logging.info("extracting data for range=%s", hist)
        data = datasource.get_times_data(self, hist.from_ts, hist.to_ts)

        # Only a subset of history will be used for computing the prediction
        X_until = None # right bound for prediction
        i = None

        for i, (_, val, timeval) in enumerate(data):
            dataset[i] = val

            dt = make_datetime(timeval)
            ts = dt.timestamp()
            if ts < period.to_ts:
                X_until = i + 1

        if i is None:
            raise errors.NoData("no data found for time range {}".format(hist))

        self.apply_defaults(dataset)

        nb_buckets_found = i + 1
        if nb_buckets_found < nb_buckets:
            dataset = np.resize(dataset, (nb_buckets_found,))

        logging.info("found %d time periods", nb_buckets_found)

        norm_dataset = self.scale_dataset(dataset)
        X_miss_val, X_test = self._format_dataset(norm_dataset[:X_until])
        if len(X_test) == 0:
            raise errors.LoudMLException("not enough data for prediction")
   
        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = self._encoder_model.predict([X_test, X_miss_val],
                                                   batch_size=g_mc_batch_size)

        if x_dim < 0 or y_dim < 0:
            mses = []
            for (x, y) in itertools.combinations(range(0,latent_dim), 2):
                _mean = np.mean(z_mean, axis=0)[[x,y]]
                mse = ((z_mean[:,[x,y]] - _mean) ** 2).mean(axis=0)
                mses.append([x, y, mse[0] + mse[1]])
    
            mses = sorted(mses, key=lambda x:x[2])
            x_dim = mses[0][0]
            y_dim = mses[0][1]

        excl = [x for x in range(latent_dim) if x != x_dim and x != y_dim]

        fig = plt.figure(figsize=(12, 10))
        if latent_dim > 3:
            ax = plt.axes(projection='3d')
            ax.set_zticks([])
        else:
            ax = plt.axes()
        
        # Hide grid lines
        ax.grid(False)
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])

        if latent_dim > 3:
            zc = np.array([
                [z_mean[i, excl[0]], z_mean[i, excl[1]], z_mean[i, excl[2]]]
                for i, _ in enumerate(z_mean)
            ])
            # (x-min(x))/(max(x)-min(x)). RGBA values should be within 0-1 range
            zc = (zc - np.min(zc, axis=0)) / (np.max(zc, axis=0) - np.min(zc, axis=0))
            if latent_dim > 5:
                ax.set_zlabel("z[{}]".format(excl[3]))
                ax.scatter(z_mean[:, x_dim], z_mean[:, y_dim], z_mean[:, excl[3]], c=zc)
            else:
                zc[:, 0] = 0
                ax.set_zlabel("z[{}]".format(excl[0]))
                ax.scatter(z_mean[:, x_dim], z_mean[:, y_dim], z_mean[:, excl[0]], c=zc)
        else:
            plt.scatter(z_mean[:, x_dim], z_mean[:, y_dim], c=z_mean[:, excl[0]])
            plt.colorbar()

        plt.xlabel("z[{}]".format(x_dim))
        plt.ylabel("z[{}]".format(y_dim))
        if output is None:
            plt.show()
        else:
            plt.savefig(output)


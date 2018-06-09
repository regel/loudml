"""
Loud ML command line tool
"""

import loudml.vendor

import argparse
import json
import logging
import pkg_resources
import os
import yaml

import loudml.config
import loudml.datasource
import loudml.model

from . import (
    errors,
)

from .errors import (
    LoudMLException,
    ModelNotTrained,
)
from .misc import (
    parse_constraint,
    make_ts,
    ts_to_str,
    get_date_ranges,
)
from .filestorage import (
    FileStorage,
)


def get_datasource(config, src_name):
    """
    Get and load data source by name
    """
    settings = config.get_datasource(src_name)
    return loudml.datasource.load_datasource(settings)


class Command:
    def __init__(self):
        self._config_path = None
        self._config = None

    def set_config(self, path):
        """
        Set path to the configuration file
        """
        self._config_path = path

    @property
    def config(self):
        if self._config is None:
            self._config = loudml.config.load_config(self._config_path)
        return self._config

    def add_args(self, parser):
        """
        Declare command arguments
        """

    def exec(self, args):
        """
        Execute command
        """


class CreateModelCommand(Command):
    """
    Create model
    """

    def add_args(self, parser):
        parser.add_argument(
            'model_file',
            help="Model file",
            type=str,
        )
        parser.add_argument(
            '-f', '--force',
            help="Overwrite if present (warning: training data will be lost!)",
            action='store_true',
        )

    def _load_model_json(self, path):
        """
        Load model JSON
        """
        with open(path) as model_file:
            return json.load(model_file)

    def _load_model_yaml(self, path):
        """
        Load model JSON
        """
        try:
            with open(path) as model_file:
                return yaml.load(model_file)
        except OSError as exn:
            raise LoudMLException(exn)
        except yaml.YAMLError as exn:
            raise LoudMLException(exn)

    def load_model_file(self, path):
        """
        Load model file
        """

        _, ext = os.path.splitext(path)
        if ext in [".yaml", ".yml"]:
            settings = self._load_model_yaml(path)
        else:
            settings = self._load_model_json(path)

        return settings

    def exec(self, args):
        model_settings = self.load_model_file(args.model_file)
        model = loudml.model.load_model(settings=model_settings,
                                        config=self.config)

        storage = FileStorage(self.config.storage['path'])

        if args.force and storage.model_exists(model.name):
            storage.delete_model(model.name)

        storage.create_model(model, self.config)
        logging.info("model '%s' created", model.name)


class ListModelsCommand(Command):
    """
    List models
    """

    def add_args(self, parser):
        parser.add_argument(
            '-i', '--info',
            help="Display model information",
            action='store_true',
        )

    def exec(self, args):
        storage = FileStorage(self.config.storage['path'])

        if args.info:
            print("MODEL                            type             trained")
            print("=========================================================")
            for name in storage.list_models():
                model = storage.load_model(name)

                print("{:32} {:16} {:3}".format(
                    name,
                    model.type,
                    'yes' if model.is_trained else 'no',
                ))
        else:
            for model in storage.list_models():
                print(model)


class DeleteModelCommand(Command):
    """
    Delete a model
    """

    def add_args(self, parser):
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )

    def exec(self, args):
        storage = FileStorage(self.config.storage['path'])
        storage.delete_model(args.model_name)
        logging.info("model '%s' deleted", args.model_name)


class ShowModelCommand(Command):
    """
    Delete a model
    """

    def add_args(self, parser):
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )
        parser.add_argument(
            '-a', '--all',
            help="All internal information",
            action='store_true',
            dest='show_all',
        )
        parser.add_argument(
            '-s', '--statistics',
            help="Give overall model statistics",
            action='store_true',
            dest='show_stats',
        )
        parser.add_argument(
            '-y', '--yaml',
            help="Dump in yaml format",
            action='store_true',
        )

    def exec(self, args):
        storage = FileStorage(self.config.storage['path'])
        model = storage.load_model(args.model_name)
        if args.show_all:
            if args.yaml:
                print(yaml.dump(model.show(), default_flow_style=False))
            else:
                print(json.dumps(model.show(), indent=4))
        elif args.show_stats:
            print(model.show(show_summary=True))
        else:
            if args.yaml:
                print(yaml.dump(model.preview, default_flow_style=False))
            else:
                print(json.dumps(model.preview, indent=4))


class TrainCommand(Command):
    """
    Train model on data set
    """

    def add_args(self, parser):
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )
        parser.add_argument(
            '-d', '--datasource',
            help="Datasource",
            type=str,
        )
        parser.add_argument(
            '-f', '--from',
            help="From date",
            type=str,
            dest='from_date',
        )
        parser.add_argument(
            '-t', '--to',
            help="To date",
            type=str,
            default="now",
            dest='to_date',
        )
        parser.add_argument(
            '-m', '--max-evals',
            help="Maximum number of training iterations",
            type=int,
        )
        parser.add_argument(
            '-e', '--epochs',
            help="Limit the number of epochs used for training",
            default=100,
            type=int,
        )
        parser.add_argument(
            '-l', '--limit',
            help="Limit the number of keys used for training",
            default=-1,
            type=int,
        )

    def exec(self, args):
        storage = FileStorage(self.config.storage['path'])
        model = storage.load_model(args.model_name)
        source = get_datasource(self.config, args.datasource or
                                model.default_datasource)

        if model.type == 'timeseries':
            if not args.from_date:
                raise LoudMLException(
                    "'from' argument is required for time-series",
                )
            if not args.to_date:
                raise LoudMLException(
                    "'to' argument is required for time-series",
                )
            result = model.train(
                source,
                args.from_date,
                args.to_date,
                max_evals=args.max_evals,
                license=self.config.license,
            )
            print("loss: %f" % result['loss'])
        elif model.type == 'fingerprints':
            if not args.from_date:
                raise LoudMLException(
                    "'from' argument is required for fingerprints",
                )
            if not args.to_date:
                raise LoudMLException(
                    "'to' argument is required for fingerprints",
                )
            model.train(
                source,
                args.from_date,
                args.to_date,
                num_epochs=args.epochs,
                limit=args.limit,
            )
        else:
            raise errors.UnsupportedModel(model.type)

        storage.save_model(model)


def _save_timeseries_prediction(
    config,
    model,
    prediction,
    source,
    datasink=None,
):
    if datasink is None:
        datasink = model.default_datasink

    if datasink is None or datasink == source.name:
        sink = source
    else:
        try:
            sink_settings = config.get_datasource(
                datasink
            )
            sink = loudml.datasource.load_datasource(sink_settings)
        except errors.LoudMLException as exn:
            logging.error("cannot load data sink: %s", str(exn))
            return

    sink.save_timeseries_prediction(prediction, model)

class ForecastCommand(Command):
    """
    Forecast the next measurements
    """

    def add_args(self, parser):
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )
        parser.add_argument(
            '-d', '--datasource',
            help="Data source",
            type=str,
        )
        parser.add_argument(
            '-c', '--constraint',
            help="Test constraint, using format: feature:low|high:threshold",
            type=str,
        )
        parser.add_argument(
            '-f', '--from',
            help="From date",
            type=str,
            dest='from_date',
        )
        parser.add_argument(
            '-t', '--to',
            help="To date",
            type=str,
            dest='to_date',
        )

        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            '-b', '--buckets',
            action='store_true',
            help="Format as buckets instead of time-series",
        )
        group.add_argument(
            '-s', '--save',
            action='store_true',
            help="Save prediction into the data source",
        )

        parser.add_argument(
            '--datasink',
            dest='datasink',
            help="name of data sink for prediction saving",
        )

    def _dump(self, data):
        """
        Dump data to stdout
        """
        print(json.dumps(data, indent=4))

    def exec(self, args):
        storage = FileStorage(self.config.storage['path'])
        model = storage.load_model(args.model_name)
        source = get_datasource(
            self.config,
            args.datasource or model.default_datasource,
        )

        if not model.is_trained:
            raise ModelNotTrained()

        if model.type == 'timeseries':
            if not args.from_date:
                raise LoudMLException(
                    "'from' argument is required for time-series")
            if not args.to_date:
                raise LoudMLException(
                    "'to' argument is required for time-series")

            constraint = parse_constraint(args.constraint) if args.constraint \
                else None

            prediction = model.forecast(
                source,
                args.from_date,
                args.to_date,
                license=self.config.license,
            )
            if constraint:
                model.test_constraint(
                    prediction,
                    constraint['feature'],
                    constraint['type'],
                    constraint['threshold'],
                )

            if args.save:
                _save_timeseries_prediction(
                    self.config,
                    model,
                    prediction,
                    source,
                    args.datasink,
                )
            else:
                if args.buckets:
                    data = prediction.format_buckets()
                else:
                    data = prediction.format_series()
                self._dump(data)


class PredictCommand(Command):
    """
    Ask model for a prediction
    """

    def add_args(self, parser):
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )
        parser.add_argument(
            '-d', '--datasource',
            help="Data source",
            type=str,
        )
        parser.add_argument(
            '-f', '--from',
            help="From date",
            type=str,
            dest='from_date',
        )
        parser.add_argument(
            '-t', '--to',
            help="To date",
            type=str,
            dest='to_date',
        )
        parser.add_argument(
            '-a', '--anomalies',
            help="Detect anomalies",
            action='store_true',
        )

        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            '-b', '--buckets',
            action='store_true',
            help="Format as buckets instead of time-series",
        )
        group.add_argument(
            '-s', '--save',
            action='store_true',
            help="Save prediction",
        )

        parser.add_argument(
            '--datasink',
            dest='datasink',
            help="name of data sink for prediction saving",
        )

    def _dump(self, data):
        """
        Dump data to stdout
        """
        print(json.dumps(data, indent=4))


    def exec(self, args):
        storage = FileStorage(self.config.storage['path'])
        model = storage.load_model(args.model_name)
        source = get_datasource(
            self.config,
            args.datasource or model.default_datasource,
        )

        if not model.is_trained:
            raise ModelNotTrained()

        if model.type == 'timeseries':
            if not args.from_date:
                raise LoudMLException(
                    "'from' argument is required for time-series")
            if not args.to_date:
                raise LoudMLException(
                    "'to' argument is required for time-series")

            if args.anomalies:
                prediction = model.predict2(
                    source,
                    args.from_date,
                    args.to_date,
                    mse_rtol=self.config.server['mse_rtol'],
                    license=self.config.license,
                )
                prediction.stat()
                model.detect_anomalies(prediction)
            else:
                prediction = model.predict(
                    source,
                    args.from_date,
                    args.to_date,
                    license=self.config.license,
                )

            if args.save:
                _save_timeseries_prediction(
                    self.config,
                    model,
                    prediction,
                    source,
                    args.datasink,
                )
            else:
                if args.buckets:
                    data = prediction.format_buckets()
                else:
                    data = prediction.format_series()
                self._dump(data)
        elif model.type == 'fingerprints':
            if not args.from_date:
                raise LoudMLException(
                    "'from' argument is required for fingerprints")
            if not args.to_date:
                raise LoudMLException(
                    "'to' argument is required for fingerprints")

            prediction = model.predict(
                source,
                args.from_date,
                args.to_date,
            )
            if args.anomalies:
                model.detect_anomalies(prediction)
            if args.save:
                storage.save_model(model)
            else:
                print(prediction)


class RunCommand(Command):
    """
    Run model predictions between 2 given dates
    """

    def add_args(self, parser):
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )
        parser.add_argument(
            '-d', '--datasource',
            help="Data source",
            type=str,
        )
        parser.add_argument(
            '-f', '--from',
            help="From date",
            type=str,
            dest='from_date',
        )
        parser.add_argument(
            '-t', '--to',
            help="To date",
            type=str,
            dest='to_date',
        )
        parser.add_argument(
            '-m', '--min_threshold',
            help="Minimum threshold for anomaly detection",
            type=int,
        )
        parser.add_argument(
            '-M', '--max_threshold',
            help="Maximum threshold for anomaly detection",
            type=int,
        )
        parser.add_argument(
            '-k', '--key',
            help="Filter with the given key only",
            type=str,
        )

    def exec(self, args):
        storage = FileStorage(self.config.storage['path'])
        model = storage.load_model(args.model_name)
        source = get_datasource(
            self.config,
            args.datasource or model.default_datasource,
        )

        if not model.is_trained:
            raise ModelNotTrained()
        
        if args.min_threshold is None:
            min_threshold = model.threshold
        else:
            min_threshold = args.min_threshold

        if args.max_threshold is None:
            max_threshold = 100
        else:
            max_threshold = args.max_threshold

        model.threshold = min_threshold

        if model.type == 'fingerprints':
            if not args.from_date:
                raise LoudMLException("'from' argument is required for fingerprints")
            if not args.to_date:
                raise LoudMLException("'to' argument is required for fingerprints")

            from_ts = make_ts(args.from_date)
            max_ts = make_ts(args.to_date)
            date_ranges = get_date_ranges(from_ts, max_ts, model.span, model.interval)
            predictions = model.predict_ranges_and_scores(
                source,
                date_ranges,
                args.key,
            )
            state = set()
            for prediction in predictions:
                for fp in prediction.fingerprints:
                    key = fp['key']
                    stats = fp['stats']
                    max_score = stats['score']
                    source.insert_times_data(
                        ts=prediction.to_ts,
                        data={ 'score': max_score },
                        tags={ model.key: key },
                        measurement='scores_{}'.format(model.name),
                    )
                    if max_score >= max_threshold and not key in state:
                        state.add(key)
                        date_str = ts_to_str(prediction.to_ts - model.interval)
                        logging.warning("detected anomaly for model '%s' and key '%s' at %s (score = %.1f)",
                                        model.name, key, date_str, max_score)

                    elif max_score >= min_threshold and key in state:
                        date_str = ts_to_str(prediction.to_ts)
                        logging.warning(
                            "anomaly still in progress for model '%s' and key '%s' at %s (score = %.1f)",
                            model.name, key, date_str, max_score,
                        )
                    elif max_score < min_threshold and key in state:
                        state.remove(key)
                        date_str = ts_to_str(prediction.from_ts)
                        logging.info(
                            "anomaly ended for model '%s' and key '%s' at %s (score = %.1f)",
                            model.name, key, date_str, max_score,
                        )
                   
                del prediction
        

def get_commands():
    """
    Get Loud ML CLI commands
    """
    for ep in pkg_resources.iter_entry_points('loudml.commands'):
        yield ep.name, ep.load()()


def main(argv=None):
    """
    Loud ML command-line tool
    """

    parser = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-c', '--config',
        help="Path to configuration file",
        type=str,
        default="/etc/loudml/config.yml",
    )
    subparsers = parser.add_subparsers(
        title="Commands",
        dest="command",
    )

    for name, command in get_commands():
        subparser = subparsers.add_parser(name)
        command.add_args(subparser)
        subparser.set_defaults(set_config=command.set_config)
        subparser.set_defaults(exec=command.exec)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    try:
        args.set_config(args.config)
        args.exec(args)
        return 0
    except LoudMLException as exn:
        logging.error(exn)
    except KeyboardInterrupt:
        logging.error("operation aborted")

    return 1

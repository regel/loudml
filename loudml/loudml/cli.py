"""
LoudML command line tool
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
    make_bool,
    parse_constraint,
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

    def set_config(self, config_path):
        """
        Set path to the configuration file
        """
        self._config_path = config_path

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
        model_settings['allowed'] = self.config.server['allowed_models']
        model = loudml.model.load_model(settings=model_settings)

        storage = FileStorage(self.config.storage['path'])

        if len(storage.list_models()) >= self.config.server['maxrunningmodels']:
            raise errors.LimitReached(
                "maximum number of running models is reached",
            )

        if args.force and storage.model_exists(model.name):
            storage.delete_model(model.name)

        storage.create_model(model)
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

    def exec(self, args):
        storage = FileStorage(self.config.storage['path'])
        model = storage.load_model(args.model_name)
        if args.show_all:
            print(json.dumps(model.show(), indent=4))
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
        source = get_datasource(self.config, args.datasource or model.default_datasource)

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
                raise LoudMLException("'from' argument is required for time-series")
            if not args.to_date:
                raise LoudMLException("'to' argument is required for time-series")

            constraint = parse_constraint(args.constraint) if args.constraint \
                         else None

            prediction = model.forecast(
                source,
                args.from_date,
                args.to_date,
            )
            if constraint:
                model.test_constraint(
                    prediction,
                    constraint['feature'],
                    constraint['type'],
                    constraint['threshold'],
                )

            if args.save:
                source.save_timeseries_prediction(prediction, model)
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
            help="Save prediction into the data source",
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
                raise LoudMLException("'from' argument is required for time-series")
            if not args.to_date:
                raise LoudMLException("'to' argument is required for time-series")

            prediction = model.predict(
                source,
                args.from_date,
                args.to_date,
            )

            if args.anomalies:
                model.detect_anomalies(prediction)
                if args.save:
                    for bucket in prediction.format_buckets():
                        stats = bucket.get('stats')
                        score = stats.get('score')
                        is_anomaly = stats.get('anomaly')
                        source.insert_times_data(
                            ts=bucket['timestamp'],
                            data={ 'score': score },
                            tags={ 'anomaly': is_anomaly },
                            measurement='scores_{}'.format(model.name),
                        )

            if args.save:
                source.save_timeseries_prediction(prediction, model)
            else:
                if args.buckets:
                    data = prediction.format_buckets()
                else:
                    data = prediction.format_series()
                self._dump(data)
        elif model.type == 'fingerprints':
            if not args.from_date:
                raise LoudMLException("'from' argument is required for fingerprints")
            if not args.to_date:
                raise LoudMLException("'to' argument is required for fingerprints")

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


def get_commands():
    """
    Get LoudML CLI commands
    """
    for ep in pkg_resources.iter_entry_points('loudml.commands'):
        yield ep.name, ep.load()()


def main(argv=None):
    """
    LoudML command-line tool
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

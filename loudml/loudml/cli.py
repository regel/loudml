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
import json

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
from .metrics import (
    send_metrics,
)
from .misc import (
    load_nab,
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


class LoadDataCommand(Command):
    """
    Load public NAB data set
    """

    def add_args(self, parser):
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
            default="now-30d",
        )

    def exec(self, args):
        if not args.datasource:
            raise LoudMLException(
                "'datasource' argument is required",
            )

        source = get_datasource(self.config, args.datasource)
        load_nab(source, args.from_date)


class LoadCheckpointCommand(Command):
    """
    Load checkpoint
    """

    def add_args(self, parser):
        parser.add_argument(
            '-c', '--checkpoint',
            help="Checkpoint name",
            type=str,
        )
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )

    def exec(self, args):
        if not args.checkpoint:
            raise LoudMLException(
                "'checkpoint' argument is required")

        storage = FileStorage(self.config.storage['path'])
        storage.set_current_ckpt(args.model_name, args.checkpoint)


class SaveCheckpointCommand(Command):
    """
    Save new checkpoint
    """

    def add_args(self, parser):
        parser.add_argument(
            '-c', '--checkpoint',
            help="Checkpoint name",
            type=str,
        )
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )

    def exec(self, args):
        if not args.checkpoint:
            raise LoudMLException(
                "'checkpoint' argument is required")

        storage = FileStorage(self.config.storage['path'])
        model = storage.load_model(args.model_name)
        storage.save_state(model, args.checkpoint)


class ListCheckpointsCommand(Command):
    """
    List checkpoints
    """

    def add_args(self, parser):
        parser.add_argument(
            '-i', '--info',
            help="Display checkpoint information",
            action='store_true',
        )
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )

    def exec(self, args):
        storage = FileStorage(self.config.storage['path'])
        if args.info:
            print("checkpoint             loss   ")
            print("==============================")
            for ckpt_name in storage.list_checkpoints(args.model_name):
                data = storage.get_model_data(args.model_name, ckpt_name)
                print("{:22} {:.5f}".format(
                    ckpt_name,
                    data['state'].get('loss'),
                ))
        else:
            for ckpt_name in storage.list_checkpoints(args.model_name):
                print(ckpt_name)


class CreateModelCommand(Command):
    """
    Create model
    """

    def add_args(self, parser):
        parser.add_argument(
            '-t', '--template',
            help="Template name",
            type=str,
        )
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
        storage = FileStorage(self.config.storage['path'])
        if args.template is not None:
            params = self._load_model_json(args.model_file)
            model = storage.load_template(
                args.template, config=self.config, **params)
        else:
            model_settings = self.load_model_file(args.model_file)
            model = loudml.model.load_model(settings=model_settings,
                                            config=self.config)

        if args.force and storage.model_exists(model.name):
            storage.delete_model(model.name)

        storage.create_model(model, self.config)
        send_metrics(self.config.metrics, storage, user_agent="loudml")
        logging.info("model '%s' created", model.name)


class ListTemplatesCommand(Command):
    """
    List templates
    """

    def exec(self, args):
        storage = FileStorage(self.config.storage['path'])

        for tmpl in storage.list_templates():
            print(tmpl)


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
        send_metrics(self.config.metrics, storage, user_agent="loudml")
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


class PlotCommand(Command):
    """
    Plot model latent space on data set
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
            '-x',
            help="Z dimension to plot on the x axis",
            type=int,
            default=-1,
        )
        parser.add_argument(
            '-y',
            help="Z dimension to plot on the y axis",
            type=int,
            default=-1,
        )
        parser.add_argument(
            '-o', '--output',
            help="Output figure to file",
            type=str,
            default=None,
        )

    def exec(self, args):
        if args.model_name == '*':
            return self.exec_all(args)

        storage = FileStorage(self.config.storage['path'])
        model = storage.load_model(args.model_name)
        source = get_datasource(self.config, args.datasource or
                                model.default_datasource)

        if model.type == 'donut':
            if not args.from_date:
                raise LoudMLException(
                    "'from' argument is required for time-series",
                )
            if not args.to_date:
                raise LoudMLException(
                    "'to' argument is required for time-series",
                )
            model.plot_results(
                source,
                args.from_date,
                args.to_date,
                num_cpus=self.config.inference['num_cpus'],
                num_gpus=self.config.inference['num_gpus'],
                x_dim=args.x,
                y_dim=args.y,
                output=args.output,
            )
        else:
            raise errors.UnsupportedModel(model.type)


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
        parser.add_argument(
            '-i', '--incremental',
            help="Resume training from the current checkpoint",
            action='store_true',
        )

    def exec_all(self, args):
        storage = FileStorage(self.config.storage['path'])
        for name in storage.list_models():
            args.model_name = name
            self.exec(args)

    def exec(self, args):
        if args.model_name == '*':
            return self.exec_all(args)

        storage = FileStorage(self.config.storage['path'])
        model = storage.load_model(args.model_name)
        source = get_datasource(self.config, args.datasource or
                                model.default_datasource)

        if model.type in ['timeseries', 'donut']:
            if not args.from_date:
                raise LoudMLException(
                    "'from' argument is required for time-series",
                )
            if not args.to_date:
                raise LoudMLException(
                    "'to' argument is required for time-series",
                )
            windows = source.list_anomalies(
                args.from_date,
                args.to_date,
                tags={'model': args.model_name},
            )
            result = model.train(
                source,
                args.from_date,
                args.to_date,
                max_evals=args.max_evals,
                num_cpus=self.config.training['num_cpus'],
                num_gpus=self.config.training['num_gpus'],
                incremental=args.incremental,
                windows=windows,
            )
            print("loss: %f" % result['loss'])
        else:
            raise errors.UnsupportedModel(model.type)

        storage.save_model(model)
        send_metrics(self.config.metrics, storage, user_agent="loudml")


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
        parser.add_argument(
            '-p',
            help="percentage of confidence interval",
            type=float,
            default=0.68,  # = +/-1 STD
            dest='p_val',
        )
        parser.add_argument(
            '-n',
            help="percentage of additional uniform noise for each 24 hours period",
            type=float,
            default=0.0,
            dest='noise_val',
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

        if model.type in ['timeseries', 'donut']:
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
                percent_interval=args.p_val,
                percent_noise=args.noise_val,
                num_cpus=self.config.inference['num_cpus'],
                num_gpus=self.config.inference['num_gpus'],
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

        send_metrics(self.config.metrics, storage, user_agent="loudml")


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
            '-k', '--key',
            help="Filter with the given key only",
            type=str,
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

        if model.type in ['timeseries', 'donut']:
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
                    num_cpus=self.config.inference['num_cpus'],
                    num_gpus=self.config.inference['num_gpus'],
                )
                prediction.stat()
                model.detect_anomalies(prediction)
            else:
                prediction = model.predict(
                    source,
                    args.from_date,
                    args.to_date,
                    num_cpus=self.config.inference['num_cpus'],
                    num_gpus=self.config.inference['num_gpus'],
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

        send_metrics(self.config.metrics, storage, user_agent="loudml")


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

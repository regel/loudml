"""
LoudML command line tool
"""

import argparse
import logging
import pkg_resources
import os
import sys
import yaml

import loudml_new.datasource
import loudml_new.model

from loudml_new.errors import (
    LoudMLException,
)
from loudml_new.filestorage import (
    FileStorage,
)

def load_config(args):
    """
    Load configuration file
    """
    try:
        with open(args.config) as config_file:
            config = yaml.load(config_file)
    except OSError as exn:
        raise LoudMLException(exn)
    except yaml.YAMLError as exn:
        raise LoudMLException(exn)

    if 'storage' not in config:
        config['storage'] = {}

    if 'path' not in config['storage']:
        config['storage']['path'] = "/var/lib/loudml"

    return config

def get_datasource(config, src_name):
    """
    Get data source by name
    """
    settings = config['datasources'].get(src_name)
    if settings is None:
        raise LoudMLException("unknown datasource '{}'".format(src_name))
    return loudml_new.datasource.load_datasource(settings)

class Command:
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

        if not settings.get('name'):
            raise LoudMLException('model has no name')
        if not settings.get('features'):
            raise LoudMLException('model has no features')

        return settings

    def exec(self, args):
        config = load_config(args)

        model_settings = self.load_model_file(args.model_file)
        model = loudml_new.model.load_model(settings=model_settings)

        storage = FileStorage(config['storage']['path'])
        storage.create_model(model)


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
        config = load_config(args)
        storage = FileStorage(config['storage']['path'])

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
        config = load_config(args)
        storage = FileStorage(config['storage']['path'])
        storage.delete_model(args.model_name)


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
            'datasource',
            help="Data source name",
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

    def exec(self, args):
        config = load_config(args)
        storage = FileStorage(config['storage']['path'])
        source = get_datasource(config, args.datasource)
        model = storage.load_model(args.model_name)

        if model.type == 'timeseries':
            if not args.from_date:
                raise LoudMLException(
                    "'from' argument is required for time-series",
                )
            if not args.to_date:
                raise LoudMLException(
                    "'to' argument is required for time-series",
                )
            model.train(source, args.from_date, args.to_date)
            storage.save_model(model)



def main():
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
    subparsers = parser.add_subparsers(title="Commands")

    for ep in pkg_resources.iter_entry_points('loudml.commands'):
        subparser = subparsers.add_parser(ep.name)
        command = ep.load()()
        command.add_args(subparser)
        subparser.set_defaults(exec=command.exec)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    args = parser.parse_args()

    try:
        args.exec(args)
    except LoudMLException as exn:
        logging.error(exn)
        sys.exit(1)

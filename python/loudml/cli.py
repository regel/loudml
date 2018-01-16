"""
LoudML command line tool
"""

import argparse
import logging
import pkg_resources
import os
import sys
import yaml

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

class CreateModelCommand:
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
        """
        Execute command
        """

        config = load_config(args)

        model_settings = self.load_model_file(args.model_file)
        model = loudml_new.model.load_model(settings=model_settings)

        storage = FileStorage(config['storage']['path'])
        storage.create_model(model)


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

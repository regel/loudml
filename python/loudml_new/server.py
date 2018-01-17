"""
LoudML server
"""

import argparse
import logging
import sys

import loudml_new.config

from flask import (
    Flask,
)
from flask_restful import (
    Api,
    Resource,
)
from loudml_new import (
    errors,
)
from loudml_new.filestorage import (
    FileStorage,
)

app = Flask(__name__, static_url_path='/static', template_folder='templates')
api = Api(app)

g_storage = None

def main():
    """
    LoudML server
    """

    global g_storage

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

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    app.logger.setLevel(logging.INFO)

    try:
        config = loudml_new.config.load_config(args.config)
        g_storage = FileStorage(config['storage']['path'])
    except errors.LoudMLException as exn:
        logging.error(exn)
        sys.exit(1)

    host, port = config['server']['listen'].split(':')
    app.run(host=host, port=int(port))

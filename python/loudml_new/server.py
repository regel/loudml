"""
LoudML server
"""

import argparse
import logging
import sys

import loudml_new.config
import loudml_new.model

from flask import (
    Flask,
    jsonify,
    request,
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

class ModelsResource(Resource):
    def get(self):
        global g_storage

        models = []

        for name in g_storage.list_models():
            model = g_storage.load_model(name)
            models.append({
                'settings': model.settings,
            })

        return jsonify(models)

    def put(self):
        global g_storage

        model = loudml_new.model.load_model(settings=request.json)

        try:
            g_storage.create_model(model)
        except errors.ModelExists as exn:
            return str(exn), 409

        return "success", 201


class ModelResource(Resource):
    def get(self, model_name):
        global g_storage

        try:
            model = g_storage.load_model(model_name)
        except errors.ModelNotFound as exn:
            return str(exn), 404

        return jsonify(model.settings)

    def delete(self, model_name):
        global g_storage

        try:
            g_storage.delete_model(model_name)
        except errors.ModelNotFound as exn:
            return str(exn), 404

        return "success"

    def post(self, model_name):
        global g_storage

        settings = request.json
        settings['name'] = model_name
        model = loudml_new.model.load_model(settings=request.json)

        try:
            g_storage.delete_model(model_name)
        except errors.ModelNotFound:
            pass

        g_storage.create_model(model)
        return "success"


api.add_resource(ModelsResource, "/models")
api.add_resource(ModelResource, "/models/<model_name>")

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

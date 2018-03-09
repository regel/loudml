"""
LoudML file storage
"""

import copy
import glob
import os
import json

from . import (
    errors,
)

from .storage import (
    Storage,
)

class FileStorage(Storage):
    """
    File storage
    """

    def __init__(self, path):
        self.path = path
        self.model_dir = os.path.join(path, 'models')

        try:
            os.makedirs(self.model_dir, exist_ok=True)
        except OSError as exn:
            raise errors.LoudMLException(str(exn))

    def model_path(self, name):
        """
        Build model path
        """
        return os.path.join(self.model_dir, "%s.lmm" % name)

    def _write_model(self, path, data):
        with open(path, 'w') as model_file:
            data = copy.deepcopy(data)
            try:
                del data['name']
            except KeyError:
                pass
            json.dump(data, model_file)

    def create_model(self, model):
        model_path = self.model_path(model.name)

        if os.path.exists(model_path):
            raise errors.ModelExists()

        self._write_model(model_path, model.data)

    def save_model(self, model):
        self._write_model(self.model_path(model.name), model.data)

    def delete_model(self, name):
        try:
            os.unlink(self.model_path(name))
        except FileNotFoundError:
            raise errors.ModelNotFound()

    def model_exists(self, name):
        return os.path.exists(self.model_path(name))

    def get_model_data(self, name):
        model_path = self.model_path(name)
        try:
            with open(model_path) as model_file:
                data = json.load(model_file)
                data['settings']['name'] = name
                return data
        except FileNotFoundError:
            raise errors.ModelNotFound()

    def set_threshold(self, name, threshold):
        data = self.get_model_data(name)
        data['settings']['threshold'] = threshold
        self._write_model(os.path.join(name), data)

    def list_models(self):
        return [
             os.path.splitext(os.path.basename(path))[0]
             for path in glob.glob(self.model_path('*'))
        ]

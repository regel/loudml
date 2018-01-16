"""
Local storage implementation for LoudML
"""

from . import (
    errors,
    ts_to_str,
)

from .storage import (
    Storage,
)

class MemStorage(Storage):
    """
    Memory storage
    """

    def __init__(self):
        self.models = {}

    def model_exists(self, name):
        return name in self.models

    def get_model_data(self, name):
        try:
            return self.models[name]
        except KeyError:
            raise errors.ModelNotFound()

    def list_models(self):
        return self.models.keys()

    def set_threshold(self, name, threshold):
        data = self.get_model_data(name)
        data['settings']['threshold'] = threshold

    def create_model(self, model):
        if model.name in self.models:
            raise errors.ModelExists()

        self.models[model.name] = model.data

    def delete_model(self, name):
        try:
            del self.models[name]
        except KeyError:
            raise errors.ModelNotFound()

    def save_model(self, model):
        self.models[model.name] = model

"""
Local storage implementation for LoudML
"""

from . import (
    errors,
    ts_to_str,
)
from .model import (
    Model,
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

    def get_model(self, name):
        try:
            data = self.models[name]
            return Model(name, data)
        except KeyError:
            raise errors.ModelNotFound()

    def set_threshold(self, name, threshold):
        model = self.get_model(name)
        model.data['threshold'] = threshold

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

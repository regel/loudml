"""
Base interface for LoudML storage
"""

import logging

from abc import (
    ABCMeta,
    abstractmethod,
)

from .misc import (
    load_hook,
)
from .model import (
    load_model,
)
from . import (
    errors,
)

class Storage(metaclass=ABCMeta):
    """
    Abstract class for LoudML storage
    """

    @abstractmethod
    def model_exists(self, name):
        """Tell if a model exists"""

    @abstractmethod
    def get_model_data(self, name):
        """Get model"""

    @abstractmethod
    def list_models(self):
        """List models"""

    @abstractmethod
    def create_model(self, model, config):
        """
        Create a model

        :param model: model
        :type  model: loudml.Model

        :param config: running configuration
        :type  config: loudml.Config
        """

    @abstractmethod
    def delete_model(self, name):
        """Delete model"""

    @abstractmethod
    def save_model(self, model):
        """Save model"""

    @abstractmethod
    def save_state(self, model):
        """Save model state"""

    def load_model(self, name):
        """Load model"""
        model_data = self.get_model_data(name)
        return load_model(**model_data)

    @abstractmethod
    def get_model_hook(self, model_name, hook_name):
        """Get model hook"""

    @abstractmethod
    def list_model_hooks(self, model_name):
        """List model hooks"""

    @abstractmethod
    def set_model_hook(self, model_name, hook_name, hook_type, config=None):
        """Set model hook"""

    @abstractmethod
    def delete_model_hook(self, model_name, hook_name):
        """Delete model hook"""

    def load_model_hook(self, model_name, hook_name, source=None):
        """Load one model hook"""

        hook_data = self.get_model_hook(model_name, hook_name)
        return load_hook(hook_name, hook_data, model_name, self, source)

    def load_model_hooks(self, model_name, source):
        """Load all model hooks"""

        hooks = []

        for hook_name in self.list_model_hooks(model_name):
            try:
                hook = self.load_model_hook(model_name, hook_name, source)
            except errors.LoudMLException as exn:
                logging.error("cannot load hook '%s/%s': %s",
                              model_name, hook_name, str(exn))
                continue

            hooks.append(hook)

        return hooks

    def set_model_object(self, model_name, key, data):
        """Save model object"""
        raise NotImplemented()

    def get_model_object(self, model_name, key):
        """Get model object"""
        raise NotImplemented()

    def delete_model_object(self, model_name, key):
        """Delete model object"""
        raise NotImplemented()

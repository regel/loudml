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
    def create_model(self, model):
        """Create model"""

    @abstractmethod
    def delete_model(self, name):
        """Delete model"""

    @abstractmethod
    def set_threshold(self, name, threshold):
        """Set model threshold"""

    @abstractmethod
    def save_model(self, model):
        """Save model"""

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

    def load_model_hook(self, model_name, hook_name):
        """Load one model hook"""

        hook_data = self.get_model_hook(model_name, hook_name)
        return load_hook(hook_name, hook_data)

    def load_model_hooks(self, model_name):
        """Load all model hooks"""

        hooks = []

        for hook_name in self.list_model_hooks(model_name):

            try:
                hook = load_model_hook(model_name, hook_name)
            except loudml.LoudMLException as exn:
                logging.error("cannot load hook '%s/%s': %s",
                              model_name, hook_name, str(exn))
                continue

            hooks.append(hook)

        return hooks

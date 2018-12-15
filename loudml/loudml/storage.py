"""
Base interface for Loud ML storage
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
    load_template,
    find_undeclared_variables,
)
from . import (
    errors,
)

class Storage(metaclass=ABCMeta):
    """
    Abstract class for Loud ML storage
    """

    @abstractmethod
    def model_exists(self, name):
        """Tell if a model exists"""

    @abstractmethod
    def get_model_data(self, name, ckpt_name=None):
        """Get model"""

    @abstractmethod
    def get_template_data(self, name):
        """Get model template"""

    @abstractmethod
    def list_models(self):
        """List models"""

    @abstractmethod
    def list_checkpoints(self, name):
        """List model checkpoints"""

    @abstractmethod
    def list_templates(self):
        """List templates"""

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
    def save_model(self, model, save_state=True):
        """Save model"""

    @abstractmethod
    def save_state(self, model, ckpt_name=None):
        """Save model state"""

    @abstractmethod
    def set_current_ckpt(self, model_name, ckpt_name):
        """Set active checkpoint"""

    def load_model(self, name, ckpt_name=None):
        """Load model"""
        model_data = self.get_model_data(name, ckpt_name)
        return load_model(**model_data)

    def load_template(self, _name, *args, **kwargs):
        """Load template"""
        model_data = self.get_template_data(_name)
        settings = model_data['settings']
        return load_template(settings=settings, *args, **kwargs)

    def find_undeclared_variables(self, name):
        """List undeclared variables in a given template"""
        model_data = self.get_template_data(name)
        settings = model_data['settings']
        return find_undeclared_variables(settings=settings)

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

    def load_model_hook(self, model, hook_name, source=None):
        """Load one model hook"""

        hook_data = self.get_model_hook(model['name'], hook_name)
        return load_hook(hook_name, hook_data, model, self, source)

    def load_model_hooks(self, model, source):
        """Load all model hooks"""

        hooks = []

        model_name = model['name']

        for hook_name in self.list_model_hooks(model_name):
            try:
                hook = self.load_model_hook(model, hook_name, source)
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

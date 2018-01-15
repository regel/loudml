"""
Base interface for LoudML storage
"""

from abc import (
    ABCMeta,
    abstractmethod,
)

from .model import (
    load_model,
)

class Storage(metaclass=ABCMeta):
    """
    Abstract class for LoudML storage
    """

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

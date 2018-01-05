"""
Base interface for LoudML data source
"""

from abc import (
    ABCMeta,
    abstractmethod,
)

class DataSource(metaclass=ABCMeta):
    """
    Abstract class for LoudML storage
    """

    @abstractmethod
    def get_times_data(
        self,
        model,
        from_date=None,
        to_date=None,
    ):
        """Get numeric data"""

    @abstractmethod
    def insert_data(self, index, data):
        """
        Insert entry into the index
        """

    @abstractmethod
    def insert_times_data(self, index, data):
        """
        Insert time-indexed entry
        """

    def commit(self):
        """
        Apply modifications
        """

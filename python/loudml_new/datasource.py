"""
Base interface for LoudML data source
"""
import datetime

from abc import (
    ABCMeta,
    abstractmethod,
)

class DataSource(metaclass=ABCMeta):
    """
    Abstract class for LoudML storage
    """

    def __init__(self):
        self._pending = []
        self._last_commit = datetime.datetime.now()

    def commit(self):
        """
        Send data
        """
        if len(self._pending) > 0:
            self.send_bulk(self._pending)
            del self._pending[:]
        self._last_commit = datetime.datetime.now()

    def _must_commit(self):
        """
        Tell if pending data must be sent to Elasticsearch
        """
        if len(self._pending) == 0:
            return False
        if len(self._pending) >= 1000:
            return True
        if (datetime.datetime.now() - self._last_commit).seconds >= 1:
            return True
        return False

    def enqueue(self, req):
        """
        Enqueue query to bulk buffer
        """
        self._pending.append(req)

        if self._must_commit():
            self.commit()

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

"""
Base interface for LoudML data source
"""
import datetime
import pkg_resources

from abc import (
    ABCMeta,
    abstractmethod,
)

from voluptuous import (
    ALLOW_EXTRA,
    All,
    Any,
    Length,
    Required,
    Schema,
)

from . import (
    errors,
    schemas,
)

class DataSource(metaclass=ABCMeta):
    """
    Abstract class for LoudML storage
    """

    SCHEMA = Schema({
        Required('name'): All(schemas.key, Length(max=256)),
        Required('type'): All(schemas.key, Length(max=256)),
    }, extra=ALLOW_EXTRA)

    def __init__(self, cfg):
        self.validate(cfg)

        self._cfg = cfg
        self._pending = []
        self._last_commit = datetime.datetime.now()

    @classmethod
    def validate(cls, cfg):
        """Validate configuration against the schema"""
        schemas.validate(cls.SCHEMA, cfg)

    @property
    def cfg(self):
        """
        Return data source configuration
        """
        return self._cfg

    @property
    def name(self):
        return self._cfg.get('name')

    def init(self, *args, **kwargs):
        pass

    def drop(self):
        pass

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
    def get_quadrant_data(
        self,
        model,
        aggregation,
        from_date=None,
        to_date=None,
        key=None,
    ):
        """Get quadrant aggregation data"""

    @abstractmethod
    def get_times_data(
        self,
        model,
        from_date=None,
        to_date=None,
    ):
        """Get numeric data"""

    @abstractmethod
    def insert_data(self, data):
        """
        Insert entry into the index
        """

    @abstractmethod
    def insert_times_data(self, ts, data, measurement, tags=None):
        """
        Insert time-indexed entry
        """

    @abstractmethod
    def save_timeseries_prediction(self, prediction):
        """
        Save time-series prediction to the datasource
        """

def load_datasource(settings):
    """
    Load datasource
    """
    src_type = settings['type']
    for ep in pkg_resources.iter_entry_points('loudml.datasources', src_type):
        if ep.name == src_type:
            return ep.load()(settings)
    raise errors.UnsupportedDataSource(src_type)

"""
Base interface for Loud ML data source
"""
import datetime

from abc import (
    ABCMeta,
    abstractmethod,
)

from voluptuous import (
    ALLOW_EXTRA,
    All,
    Any,
    Length,
    Optional,
    Range,
    Required,
    Schema,
)

from . import (
    errors,
    misc,
    schemas,
)

class DataSource(metaclass=ABCMeta):
    """
    Abstract class for Loud ML storage
    """

    SCHEMA = Schema({
        Required('name'): All(schemas.key, Length(max=256)),
        Required('type'): All(schemas.key, Length(max=256)),
        Optional('max_series_per_request', default=2000): All(
            int,
            Range(min=1),
        ),
    }, extra=ALLOW_EXTRA)

    def __init__(self, cfg):
        self._cfg = self.validate(cfg)
        self._pending = []
        self._last_commit = datetime.datetime.now()

    @classmethod
    def validate(cls, cfg):
        """Validate configuration against the schema"""
        return schemas.validate(cls.SCHEMA, cfg)

    @property
    def cfg(self):
        """
        Return data source configuration
        """
        return self._cfg

    @property
    def name(self):
        return self._cfg.get('name')

    @property
    def max_series_per_request(self):
        return self._cfg['max_series_per_request']

    def init(self, *args, **kwargs):
        pass

    def drop(self):
        pass

    def nb_pending(self):
        return len(self._pending)

    def clear_pending(self):
        del self._pending[:]

    def commit(self):
        """
        Send data
        """
        if self.nb_pending() > 0:
            self.send_bulk(self._pending)
            self.clear_pending()
        self._last_commit = datetime.datetime.now()

    def must_commit(self):
        """
        Tell if pending data must be sent to the datasource
        """
        nb_pending = self.nb_pending()

        if nb_pending == 0:
            return False
        if nb_pending >= 1000:
            return True
        if (datetime.datetime.now() - self._last_commit).seconds >= 1:
            return True
        return False

    def enqueue(self, req):
        """
        Enqueue query to bulk buffer
        """
        self._pending.append(req)

        if self.must_commit():
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
    def insert_times_data(
        self,
        ts,
        data,
        tags=None,
        *args,
        **kwargs
    ):
        """
        Insert time-indexed entry
        """

    @abstractmethod
    def save_timeseries_prediction(self, prediction):
        """
        Save time-series prediction to the datasource
        """

    def insert_annotation(
        self,
        dt,
        desc,
        _type,
        _id,
        measurement='annotations',
        tags=None,
    ):
        """
        Insert annotation and return data points saved to the TSDB
        """
        return None

    def update_annotation(
        self,
        dt,
        points,
    ):
        """
        Update annotation in the TSDB
        """
        return None

    def get_top_abnormal_keys(
        self,
        model,
        from_date,
        to_date,
        size=10,
    ):
        raise NotImplemented()

    def list_anomalies(
        self,
        from_date,
        to_date,
        tags=None,
    ):
        return []

def load_datasource(settings):
    """
    Load datasource
    """
    src_type = settings['type']

    datasource_cls = misc.load_entry_point('loudml.datasources', src_type)
    if datasource_cls is None:
        raise errors.UnsupportedDataSource(src_type)
    return datasource_cls(settings)

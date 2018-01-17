"""
LoudML errors
"""

class LoudMLException(Exception):
    """LoudML exception"""

    def __init__(self, msg=None):
        super().__init__(msg or self.__doc__)

class DataSourceNotFound(LoudMLException):
    """Data source not found"""

class ModelExists(LoudMLException):
    """Model exists"""

class ModelNotFound(LoudMLException):
    """Model not found"""

class ModelNotTrained(LoudMLException):
    """Model not trained"""

class UnsupportedDataSource(LoudMLException):
    """Unsupported data source"""

class UnsupportedMetric(LoudMLException):
    """Unsupported metric"""

class UnsupportedModel(LoudMLException):
    """Unsupported model"""

class NoData(LoudMLException):
    """No data"""

class TransportError(LoudMLException):
    """Transport error"""

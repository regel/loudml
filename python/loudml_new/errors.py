"""
LoudML errors
"""

class LoudMLException(Exception):
    """LoudML exception"""

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

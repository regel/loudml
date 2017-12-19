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

class UnsupportedMetric(LoudMLException):
    """Unsupported metric"""

class NoData(LoudMLException):
    """No data"""

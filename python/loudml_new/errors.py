"""
LoudML errors
"""

class LoudMLException(Exception):
    """LoudML exception"""

class ModelExists(LoudMLException):
    """Model exists"""

class ModelNotFound(LoudMLException):
    """Model not found"""

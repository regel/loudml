"""
LoudML errors
"""

class LoudMLException(Exception):
    """LoudML exception"""

    def __init__(self, msg=None):
        super().__init__(msg or self.__doc__)

class DataSourceError(LoudMLException):
    """Error occured on data source query"""

    def __init__(self, datasource, error=None):
        self.datasource = datasource
        self.error = error or self.__doc__

    def __str__(self):
        return "datasource[{}]: {}".format(self.datasource, self.error)

class DataSourceNotFound(LoudMLException):
    """Data source not found"""

class Invalid(LoudMLException):
    """Data is invalid"""

    def __init__(self, error, path=None, hint=None):
        self.error = error
        self.path = path
        self.hint = hint

    def __str__(self):
        hint = "" if self.hint is None else " ({})".format(self.hint)

        if self.path is None or len(self.path) == 0:
            return "data is invalid: {}{}".format(self.error, hint)
        else:
            path = '.'.join([str(key) for key in self.path])
            return "invalid field {}: {}{}".format(path, self.error, hint)

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

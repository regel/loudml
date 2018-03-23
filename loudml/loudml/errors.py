"""
LoudML errors
"""

class LoudMLException(Exception):
    """LoudML exception"""
    code = 500

    def __init__(self, msg=None):
        super().__init__(msg or self.__doc__)


class Conflict(LoudMLException):
    """Conflict"""
    code = 409

class DataSourceError(LoudMLException):
    """Error occured on data source query"""
    code = 500

    def __init__(self, datasource, error=None):
        self.datasource = datasource
        self.error = error or self.__doc__

    def __str__(self):
        return "datasource[{}]: {}".format(self.datasource, self.error)

class DataSourceNotFound(DataSourceError):
    """Data source not found"""
    code = 404

    def __str__(self):
        return "{} (name = '{}')".format(self.error, self.datasource)

class Invalid(LoudMLException):
    """Data is invalid"""
    code = 400

    def __init__(self, error, name=None, path=None, hint=None):
        self.error = error
        self.name = name
        self.path = path
        self.hint = hint

    def __str__(self):
        hint = "" if self.hint is None else " ({})".format(self.hint)

        if self.path is None or len(self.path) == 0:
            return "{} is invalid: {}{}".format(
                self.name or "data",
                self.error,
                hint,
            )
        else:
            path = '.'.join([str(key) for key in self.path])
            return "invalid field {}: {}{}".format(path, self.error, hint)

class LimitReached(LoudMLException):
    """Limit reached"""
    code = 429

class ModelExists(LoudMLException):
    """Model exists"""
    code = 409

class ModelNotFound(LoudMLException):
    """Model not found"""
    code = 404

class ModelNotTrained(LoudMLException):
    """Model not trained"""
    code = 412

class UnsupportedDataSource(LoudMLException):
    """Unsupported data source"""
    code = 501

    def __init__(self, datasource_type, error=None):
        self.datasource_type = datasource_type
        self.error = error or self.__doc__

    def __str__(self):
        return "{} (type = '{}')".format(self.error, self.datasource_type)


class UnsupportedMetric(LoudMLException):
    """Unsupported metric"""
    code = 501

    def __init__(self, metric, error=None):
        self.metric = metric
        self.error = error or self.__doc__

    def __str__(self):
        return "{} (type = '{}')".format(self.error, self.metric)


class UnsupportedModel(LoudMLException):
    """Unsupported model"""
    code = 501

    def __init__(self, model_type, error=None):
        self.model_type = model_type
        self.error = error or self.__doc__

    def __str__(self):
        return "{} (type = '{}')".format(self.error, self.model_type)


class NotFound(LoudMLException):
    """Not found"""
    code = 404

class NoData(NotFound):
    """No data"""

class TransportError(LoudMLException):
    """Transport error"""
    code = 503

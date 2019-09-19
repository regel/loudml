"""
Loud ML errors
"""


class LoudMLException(Exception):
    """Loud ML exception"""
    code = 500

    def __init__(self, msg=None):
        super().__init__(msg or self.__doc__)


class Conflict(LoudMLException):
    """Conflict"""
    code = 409


class BucketError(LoudMLException):
    """Error occured on bucket query"""
    code = 500

    def __init__(self, bucket, error=None):
        self.bucket = bucket
        self.error = error or self.__doc__

    def __str__(self):
        return "bucket[{}]: {}".format(self.bucket, self.error)


class BucketNotFound(BucketError):
    """Bucket not found"""
    code = 404

    def __str__(self):
        return "{} (name = '{}')".format(self.error, self.bucket)


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

    def __init__(self, name=None, version=None):
        self.name = name
        self.version = version

    def __str__(self):
        if self.version and self.name:
            name = " ({} version {})".format(self.name, self.version)
        else:
            name = "" if self.name is None else " ({})".format(self.name)
        return "Model{} not found".format(name)


class ModelNotTrained(LoudMLException):
    """Model not trained"""
    code = 400


class UnsupportedBucket(LoudMLException):
    """Unsupported bucket type"""
    code = 501

    def __init__(self, bucket_type, error=None):
        self.bucket_type = bucket_type
        self.error = error or self.__doc__

    def __str__(self):
        return "{} (type = '{}')".format(self.error, self.bucket_type)


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


class Forbidden(LoudMLException):
    """Forbidden"""
    code = 403


class NotFound(LoudMLException):
    """Not found"""
    code = 404


class NoData(NotFound):
    """No data"""


class TransportError(LoudMLException):
    """Transport error"""
    code = 503

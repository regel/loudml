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
    """
    Exception raised when LML returns a non-OK (>=400) HTTP status code.
    Or when an actual connection error happens; in that case the
    ``status_code`` will be set to ``'N/A'``.
    """
    code = 503

    @property
    def status_code(self):
        """
        The HTTP status code of the response that precipitated the error or
        ``'N/A'`` if not applicable.
        """
        return self.args[0]

    @property
    def error(self):
        """ A string error message. """
        return self.args[1]

    @property
    def info(self):
        """
        Dict of returned error info from LML, where available, underlying
        exception when not.
        """
        return self.args[2]

    def __str__(self):
        cause = ''
        try:
            if self.info:
                cause = ', %r' % self.info['error']['root_cause'][0]['reason']
        except LookupError:
            pass
        return '%s(%s, %r%s)' % (
            self.__class__.__name__, self.status_code, self.error, cause)


class ConnectionError(TransportError):
    """
    Error raised when there was an exception while talking to LML. Original
    exception from the underlying :class:`~elasticsearch.Connection`
    implementation is available as ``.info.``
    """
    def __str__(self):
        return 'ConnectionError(%s) caused by: %s(%s)' % (
            self.error, self.info.__class__.__name__, self.info)


class SSLError(ConnectionError):
    """ Error raised when encountering SSL errors. """


class ConnectionTimeout(ConnectionError):
    """ A network timeout. Doesn't cause a node retry by default. """
    def __str__(self):
        return 'ConnectionTimeout caused by - %s(%s)' % (
            self.info.__class__.__name__, self.info)

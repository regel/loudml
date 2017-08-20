class BonsaiException(Exception):
    """ exception"""

    def __init__(self, msg, code=500):
        super().__init__(msg)
        self.code = 500

class StorageException(BonsaiException):
    """Storage exception"""
    def __init__(self, msg="Query to database failed", code=500):
        super().__init__(msg, code=code)

class AuthenticationError(BonsaiException):
    """Authentication error"""

    def __init__(self, msg="Authentication error", code=401):
        super().__init__(msg, code=code)

"""
Common schemas for user input validation
"""

import loudml.errors

from voluptuous import (
    All,
    Any,
    error,
    Invalid,
    Length,
    Match,
    Range,
    Schema,
)

from .misc import (
    parse_timedelta,
)

key = All(
   str,
   Length(min=1),
   Match("^[a-zA-Z0-9-_]+$"),
)

class TimeDelta:
    """
    Schema for time-delta
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __call__(self, v):
        return parse_timedelta(v, **self._kwargs)


def validate(schema, data):
    """
    Validate data against a schema
    """

    try:
        schema(data)
    except Invalid as exn:
        raise loudml.errors.Invalid(exn.error_message, path=exn.path)

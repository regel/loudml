"""
Common schemas for user input validation
"""

import loudml.errors

from voluptuous import (
    All,
    Any,
    Boolean,
    error,
    Invalid,
    Length,
    Match,
    Optional,
    Range,
    Schema,
)

from .misc import (
    parse_timedelta,
)

key = All(
   str,
   Length(min=1),
   Match("^[a-zA-Z0-9-_@]+$"),
)

seasonality = Schema({
    Optional('daytime', default=False): Boolean(),
    Optional('weekday', default=False): Boolean(),
})

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
        return schema(data)
    except Invalid as exn:
        raise loudml.errors.Invalid(exn.error_message, path=exn.path)

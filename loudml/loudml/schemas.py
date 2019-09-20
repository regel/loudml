"""
Common schemas for user input validation
"""

import loudml.errors

from voluptuous import (
    All,
    Any,
    Boolean,
    Invalid,
    Length,
    Match,
    message,
    Optional,
    Range,
    Schema,
)

from .misc import (
    make_ts,
    parse_timedelta,
)

key = All(
    str,
    Length(min=1),
    Match("^[a-zA-Z0-9-_@]+$"),
)

dotted_key = All(
    str,
    Length(min=1),
    Match("^[a-zA-Z0-9-_@.]+$"),
)

bracket_key = All(
    str,
    Length(min=1),
    Match("^{{[a-zA-Z0-9-_@.]+}}$"),
)

seasonality = Schema({
    Optional('daytime', default=False): Boolean(),
    Optional('weekday', default=False): Boolean(),
})

score = Any(All(Any(int, float), Range(min=0, max=100)), None)


class TimeDelta:
    """
    Schema for time-delta
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __call__(self, v):
        parse_timedelta(v, **self._kwargs)
        return v


@message('expected absolute or relative date', cls=Invalid)
def Timestamp(v):
    """
    Schema for timestamps
    """

    try:
        make_ts(v)
    except TypeError:
        raise ValueError("value expected")
    return v


def validate(schema, data, name=None):
    """
    Validate data against a schema
    """

    try:
        return schema(data)
    except Invalid as exn:
        raise loudml.errors.Invalid(
            exn.error_message,
            name=name,
            path=exn.path,
        )

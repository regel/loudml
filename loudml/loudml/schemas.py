"""
Common schemas for user input validation
"""

import loudml.errors

from voluptuous import (
    ALLOW_EXTRA,
    All,
    Any,
    Boolean,
    Invalid,
    Length,
    Match,
    message,
    Required,
    Optional,
    Range,
    Schema,
)
import voluptuous as vol
from urllib.parse import urlparse

from .misc import (
    make_ts,
    parse_timedelta,
)

key = All(
    str,
    Length(min=1),
    Match("^[a-zA-Z0-9-_@]+$"),
)

time_str_key = All(
    str,
    Length(min=1),
    Match("^[:0-9]+$"),
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


class Url:
    """Validate an URL."""
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __call__(self, v):
        url_in = str(v)
        res = urlparse(url_in)
        if len(res.fragment) or len(res.query) or len(res.scheme):
            raise vol.Invalid(
                'You have attempted to access a restricted URL, the URL contains invalid data.')  # noqa
        if not len(res.path) or res.path[0] != '/':
            raise vol.Invalid(
                'You have attempted to access a restricted URL, the URL contains invalid path.')  # noqa

        return res.path


ScheduledJob = Schema({
    Required('name'): All(key, Length(max=256)),
    Required('method'): Any('head', 'get', 'post', 'patch', 'delete'),
    Required('relative_url'): All(str, Url()),
    Optional('params'): Schema({
    }, extra=ALLOW_EXTRA),
    Optional('json'): Schema({
    }, extra=ALLOW_EXTRA),
    Required('every'): Schema({
        Required('count'): Any(int, float),
        Required('unit'): Any(
            'second',
            'seconds',
            'minute',
            'minutes',
            'hour',
            'hours',
            'day',
            'days',
            'week',
            'weeks',
            'monday',
            'tuesday',
            'wednesday',
            'thursday',
            'friday',
            'saturday',
            'sunday',
            ),
        Optional('at'): All(time_str_key, Length(max=256)),
    }),
})


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

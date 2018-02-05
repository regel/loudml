"""
Miscelaneous Loud ML helpers
"""

import datetime
import dateutil.parser

from . import (
    errors,
)

def parse_timedelta(
    delta,
    min=None,
    max=None,
    min_included=True,
    max_included=True,
):
    """
    Parse time delta
    """

    if isinstance(delta, str):
        unit = {
            's': 'seconds',
            'm': 'minutes',
            'h': 'hours',
            'd': 'days',
            'w': 'weeks',
        }.get(delta[-1])

        if unit is None:
            value = delta
            unit = 'seconds'
        else:
            value = delta[:-1]
    else:
        unit = 'seconds'
        value = delta

    try:
        value = float(value)
    except ValueError:
        raise errors.Invalid("invalid time delta")

    message = "time delta must be {} {} seconds"

    if min is not None:
        if min_included:
            if value < min:
                raise errors.Invalid(message.format(">=", min))
        else:
            if value <= min:
                raise errors.Invalid(message.format(">", min))

    if max is not None:
        if max_included:
            if value > max:
                raise errors.Invalid(message.format("<=", max))
        else:
            if value >= max:
                raise errors.Invalid(message.format("<", max))

    return datetime.timedelta(**{unit: value})

def ts_to_datetime(ts):
    """
    Convert timestamp to datetime
    """
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)

def ts_to_str(ts):
    """
    Convert timestamp to string
    """
    return datetime_to_str(ts_to_datetime(ts))

def str_to_datetime(string):
    """
    Convert string (ISO or relative) to timestamp
    """
    if string.startswith("now"):
        now = datetime.datetime.now()
        if len(string) == 3:
            return now
        return now + parse_timedelta(string[3:])
    else:
        return dateutil.parser.parse(string)

def str_to_ts(string):
    """
    Convert string to timestamp
    """
    return str_to_datetime(string).timestamp()

def make_datetime(mixed):
    """
    Build a datetime object from a mixed input (second timestamp or string)
    """

    try:
        return ts_to_datetime(float(mixed))
    except ValueError:
        return str_to_datetime(mixed)

def make_ts(mixed):
    """
    Build a timestamp from a mixed input (second timestamp or ISO string or relative)
    """

    try:
        return float(mixed)
    except ValueError:
        return str_to_ts(mixed)

def datetime_to_str(dt):
    """
    Convert datetime to string
    """
    return  "%s.%03dZ" % (dt.strftime("%Y-%m-%dT%H:%M:%S"), dt.microsecond / 1000)

def ts_to_str(ts):
    """
    Convert timestamp to string
    """
    return datetime_to_str(ts_to_datetime(ts))

def parse_addr(addr, default_port=None):
    addr = addr.split(':')
    return {
        'host': 'localhost' if len(addr[0]) == 0 else addr[0],
        'port': default_port if len(addr) == 1 else int(addr[1]),
    }

def make_bool(mixed):
    """
    Convert value to boolean
    """

    if mixed is None:
        return False
    if isinstance(mixed, bool):
        return mixed

    try:
        return int(mixed) != 0
    except ValueError:
        pass

    if isinstance(mixed, str):
        mixed = mixed.lower()
        if mixed in ['', 'false', 'no']:
            return False
        if mixed in ['true', 'yes']:
            return True

    raise ValueError

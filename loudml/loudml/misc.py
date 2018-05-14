"""
Miscelaneous Loud ML helpers
"""

import datetime
import dateutil.parser
import pkg_resources
import sys

from collections import (
    Set,
    Mapping,
    deque,
)
from numbers import Number

from . import (
    errors,
)

QUOTE_ESCAPE_TRANS = str.maketrans({
    "'": "\\'",
})

DOUBLEQUOTE_ESCAPE_TRANS = str.maketrans({
    '"': '\\"',
})

def escape_quotes(string):
    """
    Escape simple quotes
    """
    return string.translate(QUOTE_ESCAPE_TRANS)

def escape_doublequotes(string):
    """
    Escaping double quotes
    """
    return string.translate(DOUBLEQUOTE_ESCAPE_TRANS)

def build_agg_name(measurement, field):
    return "agg_%s-%s" % (measurement, field)

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

    if isinstance(delta, str) and len(delta) > 0:
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

def dt_get_daytime(dt):
    """
    Return daytime of a datetime
    """
    return (dt.timestamp() / 3600) % 24

def dt_get_weekday(dt):
    """
    Return weekday of a datetime
    """
    return dt.isoweekday()

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

def deepsizeof(obj_0):
    """
    Compute object size recursively
    """
    def inner(obj, _seen_ids=set()):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, (str, bytes, Number, range, bytearray)):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, 'items'):
            size += sum(inner(k) + inner(v) for k, v in obj.items())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)

def load_entry_point(namespace, name):
    """
    Load pkg_resource entry point
    """

    for ep in pkg_resources.iter_entry_points(namespace, name):
        if ep.name == name:
            return ep.load()
    return None

def load_hook(hook_name, hook_data):
    hook_type = hook_data.get('type')
    hook_cls = load_entry_point('loudml.hooks', hook_type)

    if hook_cls is None:
        raise errors.NotFound("unknown hook type '{}'".format(hook_type))

    return hook_cls(hook_name, hook_data.get('config'))

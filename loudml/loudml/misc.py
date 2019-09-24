"""
Miscelaneous Loud ML helpers
"""

import datetime
import dateutil.parser
import hashlib
import json
import numpy as np
import pkg_resources
import math

import itertools

from uuid import getnode
from jinja2 import Environment, meta

from . import (
    errors,
)

QUOTE_ESCAPE_TRANS = str.maketrans({
    "'": "\\'",
})

DOUBLEQUOTE_ESCAPE_TRANS = str.maketrans({
    '"': '\\"',
})


def clear_fields(obj, fields, include_fields):
    if include_fields:
        out = {
            key: obj.get(key)
            for key in set(fields)
        }
        obj.clear()
        obj.update(out)
    else:
        out = {
            key: obj.get(key)
            for key in (set(obj.keys()) - set(fields))
        }
        obj.clear()
        obj.update(out)


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
        unit = delta[-1]

        if unit in '0123456789':
            unit = 's'
            value = delta
        else:
            value = delta[:-1]
    else:
        unit = 's'
        value = delta

    try:
        value = float(value)
    except ValueError:
        raise errors.Invalid("invalid time delta value")

    if unit == 'M':
        value *= 30
        unit = 'd'
    elif unit == 'y':
        value *= 365
        unit = 'd'

    unit = {
        's': 'seconds',
        'm': 'minutes',
        'h': 'hours',
        'd': 'days',
        'w': 'weeks',
    }.get(unit)

    if unit is None:
        raise errors.Invalid("invalid time delta unit")

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
    except ValueError as exn:
        if isinstance(mixed, str):
            return str_to_datetime(mixed)
        else:
            raise exn


def make_ts(mixed):
    """
    Build a timestamp from a mixed input
        (second timestamp or ISO string or relative time)
    """

    try:
        return float(mixed)
    except ValueError:
        return str_to_ts(mixed)


def datetime_to_str(dt):
    """
    Convert datetime to string
    """
    return "%s.%03dZ" % (
        dt.strftime("%Y-%m-%dT%H:%M:%S"), dt.microsecond / 1000)


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


class DateRange:
    def __init__(self, from_date, to_date):
        self.from_ts = make_ts(from_date)
        self.to_ts = make_ts(to_date)

        if self.to_ts < self.from_ts:
            raise errors.Invalid("invalid date range: {}".format(self))

    @classmethod
    def build_date_range(cls, from_date, to_date, bucket_interval):
        """
        Fixup date range to be sure that is a multiple of bucket_interval

        return timestamps
        """

        from_ts = make_ts(from_date)
        to_ts = make_ts(to_date)

        from_ts = math.floor(
            from_ts / bucket_interval) * bucket_interval
        to_ts = math.ceil(to_ts / bucket_interval) * bucket_interval

        return cls(from_ts, to_ts)

    @property
    def from_str(self):
        return ts_to_str(self.from_ts)

    @property
    def to_str(self):
        return ts_to_str(self.to_ts)

    def __str__(self):
        return "{}-{}".format(
            self.from_str,
            self.to_str,
        )


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


def get_date_ranges(from_ts, max_ts, span, interval):
    while (from_ts + span) < max_ts:
        to_ts = from_ts + span
        yield ts_to_str(from_ts), ts_to_str(to_ts)
        from_ts += interval


def load_entry_point(namespace, name):
    """
    Load pkg_resource entry point
    """

    for ep in pkg_resources.iter_entry_points(namespace, name):
        if ep.name == name:
            return ep.load()
    return None


def load_hook(hook_name, hook_data, model, storage, source):
    hook_type = hook_data.get('type')
    hook_cls = load_entry_point('loudml.hooks', hook_type)

    if hook_cls is None:
        raise errors.NotFound("unknown hook type '{}'".format(hook_type))

    return hook_cls(
        hook_name,
        hook_data.get('config'),
        model,
        storage,
        source,
    )


def parse_constraint(constraint):
    try:
        feature, _type, threshold = constraint.split(':')
    except ValueError:
        raise errors.Invalid("invalid format for 'constraint' parameter")

    if _type not in ('low', 'high'):
        raise errors.Invalid(
            "invalid threshold type for 'constraint' parameter")

    try:
        threshold = float(threshold)
    except ValueError:
        raise errors.Invalid("invalid threshold for 'constraint' parameter")

    return {
        'feature': feature,
        'type': _type,
        'threshold': threshold,
    }


#http://stackoverflow.com/questions/4284991/parsing-nested-parentheses-in-python-grab-content-by-level  # noqa
def parse_expression(string):
    """Generate parenthesized contents in string as pairs (level, contents)."""
    stack = []
    for i, c in enumerate(string):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            start = stack.pop()
            yield (len(stack), string[start + 1: i])


def nan_to_none(x):
    """
    Convert value to None if its NaN
    """
    return None if x is np.nan or np.isnan(x) else x


def list_from_np(array):
    """
    Convert numpy array into a jsonifiable list
    """
    return [nan_to_none(x) for x in array]


def hash_dict(data):
    ctx = hashlib.sha1()
    ctx.update(json.dumps(data, sort_keys=True).encode("utf-8"))
    return ctx.hexdigest()


def chunks(iterable, size=1):
    iterator = iter(iterable)
    for first in iterator:    # stops when iterator is depleted
        def chunk():          # construct generator for next chunk
            yield first       # yield element from for loop
            for more in itertools.islice(iterator, size - 1):
                yield more    # yield more elements from the iterator
        yield chunk()         # in outer generator, yield next chunk


def my_host_id():
    """
    Compute host identifier.

    Identifier is based on:
    - identifier computed by Python uuid library (usually MAC address)
    - MD5 hashing

    It is NOT based on:
    - system UUID in DMI entries (requires root privileges and may not be
      avalaible)
    - root filesystem UUID (requires root privileges)
    """

    m = hashlib.md5()
    m.update(str(getnode()).encode('ascii'))

    return m.hexdigest()


def find_undeclared_variables(settings):
    env = Environment()
    ast = env.parse(json.dumps(settings))
    return meta.find_undeclared_variables(ast)

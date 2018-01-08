"""
Loud ML
"""

import datetime
import dateutil.parser

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
    Convert string to timestamp
    """
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
    Build a timestamp from a mixed input (second timestamp or string)
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

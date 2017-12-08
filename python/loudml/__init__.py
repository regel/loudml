"""
Loud ML
"""

import datetime
import dateutil

def ts_to_datetime(ts_ms):
    """
    Convert millisecond timestamp to datetime
    """
    return datetime.datetime.fromtimestamp(ts_ms / 1000.0, datetime.timezone.utc)

def make_datetime(val):
    """
    Build a datetime object
    """

    try:
        return ts_to_datetime(int(val))
    except ValueError:
        return dateutil.parser.parse(val)

def datetime_to_str(ts):
    """
    Convert datetime to string
    """
    return  "%s.%3dZ" % (ts.strftime("%Y-%m-%dT%H:%M:%S"), ts.microsecond / 1000)

def ts_to_str(ts_ms):
    """
    Convert millisecond timestamp to string
    """
    return datetime_to_str(ts_to_datetime(ts_ms))

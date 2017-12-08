"""
Loud ML
"""

import datetime
import dateutil

def ts_to_datetime(ts):
    """
    Convert timestamp to datetime
    """
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)

def make_datetime(val):
    """
    Build a datetime object
    """

    try:
        return ts_to_datetime(int(val))
    except ValueError:
        return dateutil.parser.parse(val)

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

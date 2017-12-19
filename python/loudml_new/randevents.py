"""
Module for generating random events
"""


import datetime
import math
import random

def day_sin_variate(ts):
    """
    Sinusoid variate function with 24h-period
    """
    t0 = datetime.datetime.fromtimestamp(ts).replace(hour=0, minute=0, second=0).timestamp()
    return math.sin(2 * math.pi * (ts - t0) / (24 * 3600))

def randfloat(lo, hi):
    """
    Return random float between `lo` and `hi`
    """
    return lo + random.random() * (hi - lo)

class EventGenerator:
    """
    Random event generator
    """

    def __init__(self, lo=0, hi=10, sigma=1):
        self.lo = lo
        self.hi = hi
        self.sigma = sigma

    def generate_ts(self, from_ts, to_ts, step=0.001):
        """
        Generate timestamps between `from_ts` and `to_ts`.
        """

        from_ms = int(from_ts * 1000)
        to_ms = int(to_ts * 1000)
        step_ms = int(step * 1000)

        for ts_ms in range(from_ms, to_ms, step_ms):
            ts = ts_ms / 1000
            avg = (self.hi - self.lo) * (1 + day_sin_variate(ts)) / 2 + self.lo
            assert avg >= 0
            nb_events = random.normalvariate(avg, self.sigma)

            if nb_events <= 0:
                continue

            p = nb_events - int(nb_events)
            extra = 1 if random.random() <= p else 0
            nb_events = int(nb_events) + extra

            for i in range(nb_events):
                yield int(ts + i * step / nb_events)

def example():
    """
    Example of EventGenerator usage
    """

    to_ts = datetime.datetime.now().timestamp()
    from_ts = to_ts - 10
    generator = EventGenerator(lo=0, hi=10, sigma=1)
    for ts in generator.generate_ts(from_ts, to_ts):
        yield {
            'timestamp': ts,
            'foo': random.random(),
        }

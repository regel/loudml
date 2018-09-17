"""
Warp10 module for LoudML
"""

import json
import logging
import loudml.vendor
import math
import numpy as np

import warp10client

from voluptuous import (
    Any,
    Optional,
    Required,
    Url,
)
from . import (
    errors,
    schemas,
)
from .datasource import DataSource
from .misc import (
    datetime_to_str,
    make_datetime,
    make_ts,
)

def check_tag(k, v):
    if type(k) is not str or type(v) is not str:
        raise errors.Invalid("warp10: tags key/value must be strings")

def check_tags(tags):
    for k, v in tags.items():
        check_tag(k, v)

def build_tags(tags=None):
    lst = ["'{}' '{}'".format(*item) for item in tags.items()] if tags \
          else []

    return "{{ {} }}".format(','.join(lst))

def metric_to_bucketizer(metric):
    if metric == 'avg':
        bucketizer = 'mean'
    else:
        bucketizer = metric
    return "bucketizer.{}".format(bucketizer)

def catch_query_error(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except warp10client.client.CallException as exn:
            raise errors.DataSourceError(self.name, str(exn))
    return wrapper

class Warp10DataSource(DataSource):
    """
    Warp10 datasource
    """

    SCHEMA = DataSource.SCHEMA.extend({
        Optional('url', default='http://localhost:8080'): Url(),
        Required('read_token'): str,
        Required('write_token'): str,
        Optional('global_prefix', default=None): Any(None, str),
    })

    def __init__(self, cfg):
        cfg['type'] = 'warp10'
        super().__init__(cfg)
        self.read_token = cfg['read_token']
        self.write_token = cfg['write_token']
        self.global_prefix = cfg.get('global_prefix')
        self.warp10 = warp10client.Warp10Client(
            warp10_api_url=cfg['url'],
            read_token=self.read_token,
            write_token=self.write_token,
        )

    def build_name(self, name):
        return "{}.{}".format(self.global_prefix, name) if self.global_prefix \
               else name

    def build_selector(self, selector, is_regexp=False):
        selector = self.build_name(selector)
        if is_regexp:
            selector = "~" + selector
        return selector

    @catch_query_error
    def drop(self, tags=None, **kwargs):
        """
        Delete database
        """
        self.warp10.delete({
            'name': self.build_selector(".*", is_regexp=True),
            'tags': tags or {},
        })

    def insert_data(self, data):
        raise NotImplemented("Warp10 is a pure time-series database")

    def insert_times_data(
        self,
        ts,
        data,
        tags=None,
        *args,
        **kwargs
    ):
        """
        Insert data
        """

        ts_us = make_ts(ts) * 1e6

        if tags:
            check_tags(tags)

        for key, value in data.items():
            metric = {
                'name': self.build_selector(key),
                'value': value,
                'position': {
                    'longitude': None,
                    'latitude': None,
                    'elevation': None,
                    'timestamp': ts_us,
                },
                'tags': tags or {},
            }
            self.enqueue(metric)

    @catch_query_error
    def send_bulk(self, metrics):
        """
        Send data to Warp10
        """
        self.warp10.set(metrics)

    def get_quadrant_data(self, **kwargs):
        raise NotImplemented()

    def build_fetch(self, feature, from_str, to_str, tags=None):
        tags = {} if tags is None else dict(tags)

        if feature.match_all:
            for tag in feature.match_all:
                k, v = tag['tag'], tag['value']
                check_tag(k, v)
                tags[k] = v

        tags_str = build_tags(tags)

        return "[\n'{}'\n'{}'\n{}\n'{}'\n'{}'\n]\nFETCH".format(
            self.read_token,
            self.build_selector(feature.field),
            tags_str,
            from_str,
            to_str,
        )

    def build_multi_fetch(self, model, from_str, to_str, tags=None):
        bucket_span = int(model.bucket_interval * 1e6)

        scripts = [
            "[\n{}\n{}\n0\n{}\n0\n]\nBUCKETIZE".format(
                self.build_fetch(
                    feature,
                    from_str,
                    to_str,
                    tags,
                ),
                metric_to_bucketizer(feature.metric),
                bucket_span,
            )
            for feature in model.features
        ]
        return "[\n{}\n]".format("\n".join(scripts))

    @catch_query_error
    def get_times_data(
        self,
        model,
        from_date,
        to_date,
        tags=None,
        **kwargs
    ):
        span = model.span * 1e6

        period = model.build_date_range(from_date, to_date)

        nb_buckets = int((period.to_ts - period.from_ts) / model.bucket_interval)
        buckets = np.full((nb_buckets, len(model.features)), np.nan, dtype=float)

        script = self.build_multi_fetch(
            model,
            period.from_str,
            period.to_str,
            tags=tags,
        )
        raw = self.warp10.exec(script)
        data = json.loads(raw)

        from_us = period.from_ts * 1e6
        to_us = period.to_ts * 1e6
        bucket_interval_us = int(model.bucket_interval * 1e6)

        has_data = False

        for i, item in enumerate(data[0]):
            if len(item) == 0:
                continue

            item = item[0]
            values = item['v']

            for ts_us, value in values:
                # XXX: Warp10 buckets are labeled with the right timestamp but LoudML
                # use the left one.
                ts_us -= bucket_interval_us

                if ts_us < from_us or ts_us >= to_us:
                    # XXX Sometimes, Warp10 returns extra buckets, skip them
                    continue

                j = math.floor((ts_us - from_us) / bucket_interval_us)
                buckets[j][i] = value
                has_data = True

        if not has_data:
            raise errors.NoData()

        result = []
        from_ts = ts = from_us  / 1e6

        for bucket in buckets:
            result.append(((ts - from_ts), list(bucket), ts))
            ts += model.bucket_interval

        return result

    def save_timeseries_prediction(self, prediction, model):
        raise NotImplemented()

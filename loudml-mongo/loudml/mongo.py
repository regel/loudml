"""
MongoDB module for LoudML
"""

import logging
import math

import numpy as np
import pymongo

from voluptuous import (
    All,
    Length,
    Optional,
    Required,
)

from . import (
    errors,
    schemas,
)
from .misc import (
    make_ts,
    parse_addr,
)
from loudml.datasource import DataSource

def _tk(key):
    return "$" + key

def _build_query(feature, timestamp_field, boundaries):
    field = feature.field
    metric = feature.metric

    group_by = _tk(timestamp_field)

    query = []

    if feature.match_all:
        match = []

        for tag in feature.match_all:
            k, v = tag['tag'], tag['value']
            match.append({k: v})

        query.append({'$match': {'$or': match}})

    if metric == "count":
        return query + [
            {'$match': {field: {'$exists': True}}},
            {'$bucket': {
                'groupBy': group_by,
                'boundaries': boundaries,
                'default': None,
                'output': {feature.name: {'$sum': 1}},
            }}
        ]

    if metric == "mean":
        metric = "avg"

    return query + [
        {'$bucket': {
            'groupBy': group_by,
            'boundaries': boundaries,
            'default': None,
            'output': {feature.name: {
                _tk(metric): _tk(field),
            }}
        }}
    ]

def catch_query_error(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except (
            pymongo.errors.PyMongoError
        ) as exn:
            raise errors.DataSourceError(self.name, str(exn))
    return wrapper

class MongoDataSource(DataSource):
    """
    MongoDB datasource
    """

    SCHEMA = DataSource.SCHEMA.extend({
        Required('addr'): str,
        Required('database'): str,
        Optional('username'): All(schemas.key, Length(max=256)),
        Optional('password'): str,
    })

    def __init__(self, cfg):
        cfg['type'] = 'mongodb'
        super().__init__(cfg)
        self._client = None
        self._db = None
        self._pending = {}
        self._nb_pending = 0

    @property
    def client(self):
        if self._client is None:
            addr = parse_addr(self.cfg['addr'], default_port=8086)
            logging.info(
                "connecting to mongodb on %s:%d, using database '%s'",
                addr['host'],
                addr['port'],
                self.cfg['database'],
            )

            self._client = pymongo.MongoClient(
                host=addr['host'],
                port=addr['port'],
                username=self.cfg.get('username'),
                password=self.cfg.get('password'),
            )

        return self._client

    @property
    def db(self):
        if self._db is None:
            self._db = self.client[self.cfg['database']]
        return self._db

    @catch_query_error
    def init(self, db=None, *args, **kwargs):
        raise NotImplemented()

    @catch_query_error
    def drop(self, db=None):
        self.client.drop_database(db or self.cfg['database'])

    def nb_pending(self):
        return self._nb_pending

    def enqueue(self, collection, request):
        if collection not in self._pending:
            self._pending[collection] = []
        self._pending[collection].append(request)
        self._nb_pending += 1

    def clear_pending(self):
        self._pending = {}

    def insert_data(
        self,
        data,
        collection='generic',
        tags=None,
    ):
        if collection is None:
            raise errors.Invalid("cannot insert data: no collection given")

        if tags is not None:
            for tag, tag_val in tags.items():
                data[tag] = tag_val

        self.enqueue(collection, pymongo.InsertOne(data))

    def insert_times_data(
        self,
        ts,
        data,
        collection=None,
        tags=None,
        timestamp_field='timestamp',
        *args,
        **kwargs
    ):
        """
        Insert data
        """

        ts = make_ts(ts)

        data = data.copy()
        data[timestamp_field] = ts
        self.insert_data(data, tags=tags, collection=collection)

    @catch_query_error
    def send_bulk(self, pending):
        """
        Send data to MongoDB
        """
        for collection, requests in pending.items():
            self.db[collection].bulk_write(requests)

    @catch_query_error
    def get_quadrant_data(
        self,
        model,
        agg,
        from_date=None,
        to_date=None,
        key=None,
        limit=0,
        offset=0,
    ):
        raise NotImplemented()

    @catch_query_error
    def get_times_data(
        self,
        model,
        from_date,
        to_date,
    ):
        bucket_interval = int(model.bucket_interval)
        timestamp_field = model.timestamp_field

        from_ts = int(math.floor(make_ts(from_date) / bucket_interval) * bucket_interval)
        to_ts = int(math.ceil(make_ts(to_date) / bucket_interval) * bucket_interval)

        boundaries = list(range(from_ts, to_ts + bucket_interval, bucket_interval))

        nb_buckets = len(boundaries)
        buckets = np.full((nb_buckets, len(model.features)), np.nan, dtype=float)

        nb_buckets_found = 0

        for i, feature in enumerate(model.features):
            metric = feature.metric
            field = feature.field

            query = _build_query(feature, timestamp_field, boundaries)
            resp = self.db[feature.collection].aggregate(query)

            for entry in resp:
                ts = entry['_id']

                if ts is None:
                    continue

                value = entry[feature.name]
                j = int((ts - from_ts) / bucket_interval)
                buckets[j][i] = value
                if j >= nb_buckets_found:
                    nb_buckets_found = j + 1

        if nb_buckets_found == 0:
            raise errors.NoData()

        result = []
        ts = from_ts

        for bucket in buckets[0:nb_buckets_found]:
            result.append((ts - from_ts, list(bucket), ts))
            ts += bucket_interval

        return result

    def save_timeseries_prediction(self, prediction, model):
        collection = "prediction_" + model.name

        logging.info("saving '%s' prediction to '%s.%s'",
                     model.name, self.cfg['database'], collection)

        for bucket in prediction.format_buckets():
            data = bucket['predicted']
            tags = model.get_tags()
            stats = bucket.get('stats', None)
            if stats is not None:
                data['score'] = float(stats.get('score'))
                tags['is_anomaly'] = stats.get('anomaly', False)

            self.insert_times_data(
                collection=collection,
                ts=bucket['timestamp'],
                tags=tags,
                data=data,
            )
        self.commit()

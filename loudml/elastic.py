"""
Elasticsearch module for Loud ML
"""

import datetime
import logging
import re

import elasticsearch.exceptions
import urllib3.exceptions

import numpy as np

from elasticsearch import (
    Elasticsearch,
    helpers,
)

from voluptuous import (
    Required,
    Optional,
    All,
    Length,
    Boolean,
    IsFile,
    Range,
)

from . import (
    errors,
    schemas,
)

from loudml.bucket import Bucket
from loudml.misc import (
    escape_quotes,
    make_ts,
    parse_addr,
)


def version(v):
    return [int(x) for x in re.sub(r'(\.0+)*$', '', v).split(".")]


def ts_to_ms(ts):
    """
    Convert second timestamp to integer millisecond timestamp
    """
    return int(ts * 1e3)


def make_ts_ms(mixed):
    """
    Build a millisecond timestamp from a mixed input
    (second timestamp or string)
    """
    return ts_to_ms(make_ts(mixed))


def _date_range_to_ms(from_date=None, to_date=None):
    """
    Convert date range to millisecond
    """
    return (
        None if from_date is None else int(make_ts(from_date) * 1000),
        None if to_date is None else int(make_ts(to_date) * 1000),
    )


def _build_match_all(match_all=None):
    """
    Build filters for search query
    """

    if match_all is None:
        return

    for condition in match_all:
        key = condition['tag']
        val = condition['value']

        if isinstance(val, bool):
            val = str(val).lower()
        elif isinstance(val, str):
            val = escape_quotes(val)

        yield {
            "match": {
                key: val
            }
        }


def _build_date_range(field, from_ms=None, to_ms=None):
    """
    Build date range for search query
    """

    date_range = {}

    if from_ms is not None:
        date_range['gte'] = from_ms
    if to_ms is not None:
        date_range['lt'] = to_ms

    if len(date_range) == 0:
        return None

    date_range['format'] = 'epoch_millis'

    return {'range': {
        field: date_range,
    }}


def _build_extended_bounds(from_ms=None, to_ms=None):
    """
    Build extended_bounds
    """
    bounds = {}

    if from_ms is not None:
        bounds['min'] = from_ms
    if to_ms is not None:
        bounds['max'] = to_ms

    return bounds


class ElasticsearchBucket(Bucket):
    """
    Elasticsearch bucket
    """

    SCHEMA = Bucket.SCHEMA.extend({
        Required('addr'): str,
        Required('index'): str,
        Optional('doc_type', default='doc'): str,
        'routing': str,
        Optional('dbuser'): All(schemas.key, Length(max=256)),
        Optional('dbuser_password'): str,
        Optional('ca_certs'): IsFile(),
        Optional('client_cert'): IsFile(),
        Optional('client_key'): IsFile(),
        Optional('use_ssl', default=False): Boolean(),
        Optional('verify_ssl', default=False): Boolean(),
        Optional('number_of_shards', default=1): All(int, Range(min=1)),
        Optional('number_of_replicas', default=0): All(int, Range(min=0)),
    })

    def __init__(self, cfg):
        cfg['type'] = 'elasticsearch'
        super().__init__(cfg)
        self._es = None
        self._touched_indices = []

    @property
    def number_of_shards(self):
        return int(self.cfg.get('number_of_shards') or 1)

    @property
    def number_of_replicas(self):
        return int(self.cfg.get('number_of_replicas') or 0)

    @property
    def addr(self):
        return self.cfg['addr']

    @property
    def index(self):
        return self.cfg['index']

    @property
    def doc_type(self):
        return self.cfg['doc_type']

    @property
    def timeout(self):
        return self.cfg.get('timeout', 30)

    @property
    def dbuser(self):
        return self.cfg.get('dbuser')

    @property
    def dbuser_password(self):
        return self.cfg.get('dbuser_password')

    @property
    def use_ssl(self):
        return self.cfg.get('use_ssl') or False

    @property
    def verify_ssl(self):
        return self.cfg.get('verify_ssl') or False

    @property
    def ca_certs(self):
        return self.cfg.get('ca_certs')

    @property
    def client_cert(self):
        return self.cfg.get('client_cert')

    @property
    def client_key(self):
        return self.cfg.get('client_key')

    @property
    def es(self):
        if self._es is None:
            addr = parse_addr(self.addr, default_port=9200)
            logging.info('connecting to elasticsearch on %s:%d',
                         addr['host'], addr['port'])
            self._es = Elasticsearch(
                [addr],
                timeout=self.timeout,
                http_auth=(
                    self.dbuser,
                    self.dbuser_password) if self.dbuser else None,
                use_ssl=self.use_ssl,
                verify_certs=self.verify_ssl,
                ca_certs=self.ca_certs,
                client_cert=self.client_cert,
                client_key=self.client_key,
            )

        # urllib3 & elasticsearch modules log exceptions, even if they are
        # caught! Disable this.
        urllib_logger = logging.getLogger('urllib3')
        urllib_logger.setLevel(logging.CRITICAL)
        es_logger = logging.getLogger('elasticsearch')
        es_logger.setLevel(logging.CRITICAL)

        return self._es

    def init(self, data_schema=None, *args, **kwargs):
        """
        Create index and write mapping
        """
        if data_schema and self.timestamp_field:
            data_schema[self.timestamp_field] = {
                "type": "date",
                "format": "epoch_millis",
            }
        if data_schema:
            info = self.es.info()
            mapping = {
                "properties": data_schema
            }
            if not self.es.indices.exists(
                index=self.index,
            ):
                params = {}
                if version(info['version']['number']) >= version('7.0.0'):
                    params['include_type_name'] = 'true'
                mappings = {
                    'mappings': {
                        self.doc_type: {
                            "properties": data_schema
                        }
                    },
                    'settings': {
                        "number_of_shards": self.number_of_shards,
                        "number_of_replicas": self.number_of_replicas,
                        "codec": "best_compression",
                    }
                }
                self.es.indices.create(
                    index=self.index,
                    body=mappings,
                    params=params,
                )
            params = {
                'allow_no_indices': 'true',
                'ignore_unavailable': 'true',
            }
            if version(info['version']['number']) >= version('7.0.0'):
                params['include_type_name'] = 'true'

            self.es.indices.put_mapping(
                doc_type=self.doc_type,
                body=mapping,
                index=self.index,
                params=params,
            )

    def drop(self, index=None):
        """
        Delete index
        """
        if index is None:
            index = self.index
        self.es.indices.delete(index, ignore=404)

    def send_bulk(self, requests):
        """
        Send data to Elasticsearch
        """
        logging.info("commit %d change(s) to elasticsearch", len(requests))

        try:
            helpers.bulk(
                self.es,
                requests,
                chunk_size=5000,
                timeout="30s",
            )
        except (
            urllib3.exceptions.HTTPError,
            elasticsearch.exceptions.TransportError,
        ) as exn:
            raise errors.TransportError(str(exn))

    def refresh(self, index=None):
        """
        Explicitely refresh index
        """

        if index is None:
            indices = self._touched_indices
            self._touched_indices = []
        else:
            indices = [index]

        for i in indices:
            self.es.indices.refresh(i)

    def get_index_name(self, index=None, timestamp=None):
        """
        Build index name
        """

        if index is None:
            index = self.index

        if '*' in index:
            if timestamp is None:
                dt = datetime.datetime.now()
            else:
                dt = datetime.datetime.fromtimestamp(timestamp)

            index = index.replace('*', dt.strftime("%Y.%m.%d"))

        return index

    def insert_data(
        self,
        data,
        index=None,
        doc_type=None,
        doc_id=None,
        timestamp=None,
    ):
        """
        Insert entry into the index
        """

        index = self.get_index_name(index, timestamp)

        req = {
            '_index': index,
            '_type': doc_type or self.doc_type,
            '_source': data,
        }

        if doc_id is not None:
            req['_id'] = doc_id

        self.enqueue(req)
        self._touched_indices.append(index)

    def insert_times_data(
        self,
        ts,
        data,
        tags=None,
        index=None,
        doc_type=None,
        doc_id=None,
        *args,
        **kwargs
    ):
        """
        Insert time-indexed entry
        """
        ts = make_ts(ts)

        data[self.timestamp_field] = ts_to_ms(ts)

        if tags is not None:
            for tag, tag_val in tags.items():
                data[tag] = tag_val

        self.insert_data(
            data,
            index=index or self.index,
            doc_type=doc_type or self.doc_type,
            doc_id=doc_id,
            timestamp=int(ts),
        )

    def search(self, body, index=None, routing=None, doc_type=None, size=0):
        """
        Send search query to Elasticsearch
        """

        if index is None:
            index = self.index

        params = {}
        if routing is not None:
            params['routing'] = routing

        try:
            return self.es.search(
                index=index,
                doc_type=doc_type or self.doc_type,
                size=size,
                body=body,
                params=params,
            )
        except elasticsearch.exceptions.TransportError as exn:
            raise errors.TransportError(str(exn))
        except urllib3.exceptions.HTTPError as exn:
            raise errors.BucketError(self.name, str(exn))

    @staticmethod
    def _build_aggs(features):
        """
        Build Elasticsearch aggregations
        """

        aggs = {}

        for feature in features:
            if feature.metric in ['mean', 'average']:
                feature.metric = 'avg'
            if feature.metric in ['std_deviation', 'variance']:
                sub_agg = 'extended_stats'
            else:
                sub_agg = 'stats'

            if feature.script:
                agg = {
                    sub_agg: {
                        "script": {
                            "lang": "painless",
                            "inline": feature.script,
                        }
                    }
                }
            elif feature.field:
                agg = {
                    sub_agg: {
                        "field": feature.field,
                    }
                }

            aggs[feature.name] = agg

        return aggs

    @classmethod
    def _build_times_query(
        cls,
        bucket_interval,
        features,
        from_ms,
        to_ms,
        timestamp_field,
    ):
        """
        Build Elasticsearch query for time-series
        """

        body = {
            "size": 0,
            "aggs": {
                "histogram": {
                    "date_histogram": {
                        "field": timestamp_field,
                        "extended_bounds": _build_extended_bounds(
                            from_ms, to_ms - 1000*bucket_interval),
                        "interval": "%ds" % bucket_interval,
                        "min_doc_count": 0,
                        "time_zone": "UTC",
                        "format": "yyyy-MM-dd'T'HH:mm:ss'Z'",  # key_as_string
                        "order": {
                            "_key": "asc"
                        }
                    },
                    "aggs": {
                    },
                }
            }
        }

        must = []

        date_range = _build_date_range(timestamp_field, from_ms, to_ms)
        if date_range is not None:
            must.append(date_range)

        for feature in features:
            match_all = _build_match_all(feature.match_all)
            for condition in match_all:
                must.append(condition)

        if len(must) > 0:
            body['query'] = {
                'bool': {
                    'must': must,
                }
            }

        aggs = cls._build_aggs(features)

        for x in sorted(aggs):
            body['aggs']['histogram']['aggs'][x] = aggs[x]

        return body

    @staticmethod
    def _get_agg_val(bucket, feature):
        """
        Get aggregation value for the bucket returned by Elasticsearch
        """
        agg_val = bucket[feature.name].get(feature.metric)

        if agg_val is None:
            logging.info(
                "missing data: field '%s', metric: '%s', bucket: %s",
                feature.field, feature.metric, bucket['key'],
            )

        return agg_val

    def get_times_data(
        self,
        bucket_interval,
        features,
        from_date=None,
        to_date=None,
    ):
        from_ms, to_ms = _date_range_to_ms(from_date, to_date)

        body = self._build_times_query(
            bucket_interval,
            features,
            from_ms=from_ms,
            to_ms=to_ms,
            timestamp_field=self.timestamp_field,
        )

        es_res = self.search(
            body,
            routing=None,
        )

        hits = es_res['hits']['total']
        if hits == 0:
            return

        # TODO: last bucket may contain incomplete data when to_date == now
        """
        now = datetime.datetime.now().timestamp()
        epoch_ms = 1000 * int(now)
        min_bound_ms = 1000 * int(now / bucket_interval) * bucket_interval
        """

        t0 = None

        for bucket in es_res['aggregations']['histogram']['buckets']:
            X = np.full(len(features), np.nan, dtype=float)
            timestamp = int(bucket['key'])
            timeval = bucket['key_as_string']

            for i, feature in enumerate(features):
                X[i] = self._get_agg_val(bucket, feature)

            # TODO: last bucket may contain incomplete data when to_date == now
            """
            try:
                # The last interval contains partial data
                if timestamp == min_bound_ms:
                    R = float(epoch_ms - min_bound_ms
                       ) / (1000 * bucket_interval)
                    X = R * X + (1-R) * X_prev
            except NameError:
                # X_prev not defined. No interleaving required.
                pass

            X_prev = X
            """

            if t0 is None:
                t0 = timestamp

            yield (timestamp - t0) / 1000, X, timeval

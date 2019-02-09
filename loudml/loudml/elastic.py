"""
Elasticsearch module for Loud ML
"""

import datetime
import logging
import math

import elasticsearch.exceptions
import urllib3.exceptions

import numpy as np

from elasticsearch import (
    Elasticsearch,
    helpers,
    TransportError,
)

from voluptuous import (
    Required,
    Optional,
    All,
    Length,
    Boolean,
    IsFile,
    Schema,
)

from . import (
    errors,
    schemas,
)

from loudml.datasource import DataSource
from loudml.misc import (
    escape_quotes,
    deepsizeof,
    make_ts,
    parse_addr,
    build_agg_name,
)

# Limit ES aggregations output to 500 MB
PARTITION_MAX_SIZE = 500 * 1024 * 1024

def ts_to_ms(ts):
    """
    Convert second timestamp to integer millisecond timestamp
    """
    return int(ts * 1e3)

def make_ts_ms(mixed):
    """
    Build a millisecond timestamp from a mixed input (second timestamp or string)
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
            val = "'{}'".format(escape_quotes(val))

        yield {
          "script": {
            "script": {
              "lang": "painless",
              "inline": "doc['{}'].value=={}".format(key, val)
            }
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

class ElasticsearchDataSource(DataSource):
    """
    Elasticsearch datasource
    """

    SCHEMA = DataSource.SCHEMA.extend({
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
    })

    def __init__(self, cfg):
        cfg['type'] = 'elasticsearch'
        super().__init__(cfg)
        self._es = None
        self._touched_indices = []

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
                http_auth=(self.dbuser, self.dbuser_password) if self.dbuser else None,
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

    def init(self, template_name=None, template=None, *args, **kwargs):
        """
        Create index and put template
        """

        if template is not None:
            self.es.indices.put_template(
                name=template_name,
                body=template,
                ignore=400,
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
        timestamp_field='timestamp',
        *args,
        **kwargs
    ):
        """
        Insert time-indexed entry
        """
        ts = make_ts(ts)

        data[timestamp_field] = ts_to_ms(ts)

        if tags is not None:
            for tag, tag_val in tags.items():
                data[tag] = tag_val

        self.insert_data(
            data,
            index=index,
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
            raise errors.DataSourceError(self.name, str(exn))

    @staticmethod
    def _build_aggs(model):
        """
        Build Elasticsearch aggregations
        """

        aggs = {}

        for feature in model.features:
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

    def get_field_cardinality(
        self,
        model,
        from_ms=None,
        to_ms=None,
    ):
        body = {
          "size": 0,
          "aggs": {
            "count": {
              "cardinality": {
                "field": model.key,
              }
            }
          }
        }

        must = []
        date_range = _build_date_range(model.timestamp_field, from_ms, to_ms)
        if date_range is not None:
            must.append(date_range)

        if len(must) > 0:
            body['query'] = {
                'bool': {
                    'must': must,
                }
            }

        es_res = self.search(
            body,
            routing=model.routing,
        )

        return int(es_res['aggregations']['count']['value'])

    @staticmethod
    def build_quadrant_aggs(model, agg):
        res = {}
        fields = [feature.field for feature in agg.features]
        for field in set(fields):
            res.update({
              build_agg_name(agg.measurement, field): {
                "extended_stats": {"field": field}
              }
            })
        return res

    @staticmethod
    def read_quadrant_aggs(key, time_buckets):
        return key, time_buckets

    @classmethod
    def _build_quadrant_query(
        cls,
        model,
        aggregation,
        from_ms=None,
        to_ms=None,
        key=None,
        partition=0,
        num_partition=1,
    ):
        body = {
            "size": 0,
            "aggs": {
                "key": {
                    "terms": {
                        "field": model.key,
                        "size": model.max_keys,
                        "collect_mode" : "breadth_first",
                        "include": {
                            "partition": partition,
                            "num_partitions": num_partition,
                        },
                    },
                    "aggs": {
                        "quadrant_data": {
                            "date_histogram": {
                                "field": model.timestamp_field,
                                "interval": "%ds" % (model.bucket_interval),
                                "min_doc_count": 0,
                                "time_zone": "UTC",
                                "format": "yyyy-MM-dd'T'HH:mm:ss'Z'", # key_as_string format
                                "extended_bounds": _build_extended_bounds(from_ms, to_ms-1),
                            },
                            "aggs": cls.build_quadrant_aggs(model, aggregation),
                        }
                    }
                }
            }
        }

        must = []

        date_range = _build_date_range(model.timestamp_field, from_ms, to_ms)
        if date_range is not None:
            must.append(date_range)

        if key is not None:
            must.append({"match": {model.key: key}})

        match_all = _build_match_all(aggregation.match_all)
        for condition in match_all:
            must.append(condition)

        if len(must) > 0:
            body['query'] = {
                "bool": {
                    "must": must
                }
            }

        return body

    def get_quadrant_data(
        self,
        model,
        aggregation,
        from_date=None,
        to_date=None,
        key=None,
    ):
        from_ms, to_ms = _date_range_to_ms(from_date, to_date)

        if key is None:
            num_series = self.get_field_cardinality(model, from_ms, to_ms)
            num_partition = math.ceil(num_series / self.max_series_per_request)
        else:
            num_partition = 1

        for partition in range(0, num_partition):
            logging.info("running aggregations for model '%s', partition %d/%d",
                         model.name, partition, num_partition)

            body = self._build_quadrant_query(
                model,
                aggregation,
                from_ms=from_ms,
                to_ms=to_ms,
                key=key,
                partition=partition,
                num_partition=num_partition,
            )

            es_res = self.search(
                body,
                routing=model.routing,
            )

            for bucket in es_res['aggregations']['key']['buckets']:
                yield self.read_quadrant_aggs(
                    bucket['key'],
                    bucket['quadrant_data']['buckets'],
                )

    @classmethod
    def _build_times_query(
        cls,
        model,
        from_ms,
        to_ms,
    ):
        """
        Build Elasticsearch query for time-series
        """

        body = {
          "size": 0,
          "aggs": {
            "histogram": {
              "date_histogram": {
                "field": model.timestamp_field,
                "extended_bounds": _build_extended_bounds(from_ms, to_ms - 1000*model.bucket_interval),
                "interval": "%ds" % model.bucket_interval,
                "min_doc_count": 0,
                "time_zone": "UTC",
                "format": "yyyy-MM-dd'T'HH:mm:ss'Z'", # key_as_string format
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

        date_range = _build_date_range(model.timestamp_field, from_ms, to_ms)
        if date_range is not None:
            must.append(date_range)

        for feature in model.features:
            match_all = _build_match_all(feature.match_all)
            for condition in match_all:
                must.append(condition)

        if len(must) > 0:
            body['query'] = {
                'bool': {
                    'must': must,
                }
            }

        aggs = cls._build_aggs(model)

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
        model,
        from_date=None,
        to_date=None,
    ):
        features = model.features

        from_ms, to_ms = _date_range_to_ms(from_date, to_date)

        body = self._build_times_query(
            model,
            from_ms=from_ms,
            to_ms=to_ms,
        )

        es_res = self.search(
            body,
            routing=model.routing,
        )

        hits = es_res['hits']['total']
        if hits == 0:
            logging.info("Aggregations for model %s: Missing data", model.name)
            return

        # TODO: last bucket may contain incomplete data when to_date == now
        """
        now = datetime.datetime.now().timestamp()
        epoch_ms = 1000 * int(now)
        min_bound_ms = 1000 * int(now / model.bucket_interval) * model.bucket_interval
        """

        t0 = None

        for bucket in es_res['aggregations']['histogram']['buckets']:
            X = np.full(model.nb_features, np.nan, dtype=float)
            timestamp = int(bucket['key'])
            timeval = bucket['key_as_string']

            for i, feature in enumerate(features):
                X[i] = self._get_agg_val(bucket, feature)

            # TODO: last bucket may contain incomplete data when to_date == now
            """
            try:
                # The last interval contains partial data
                if timestamp == min_bound_ms:
                    R = float(epoch_ms - min_bound_ms) / (1000 * model.bucket_interval)
                    X = R * X + (1-R) * X_prev
            except NameError:
                # X_prev not defined. No interleaving required.
                pass

            X_prev = X
            """

            if t0 is None:
                t0 = timestamp

            yield (timestamp - t0) / 1000, X, timeval

    def gen_template(
        self,
        model,
        prediction,
    ):
        template = {
          "template": self.index,
          "mappings": {
            self.doc_type: {
              "properties": {
                "timestamp": {"type": "date", "format": "epoch_millis"},
                "score": {"type": "float"},
                "is_anomaly": {"type": "boolean"},
              }
            }
          }
        }
        properties = {}
        for tag in model.get_tags():
            properties[tag] = {"type": "keyword"}

        for field in prediction.get_field_names():
            properties[field] = {"type": "float"}

        if model.timestamp_field is not None:
            properties[model.timestamp_field] = {
                "type": "date",
                "format": "epoch_millis",
            }
        template['mappings'][self.doc_type]['properties'].update(properties)
        return template

    def save_timeseries_prediction(
        self,
        prediction,
        model,
        index=None,
    ):
        template = self.gen_template(model, prediction)
        self.init(template_name=self.index, template=template)

        for bucket in prediction.format_buckets():
            data = bucket['predicted']
            tags = model.get_tags()
            stats = bucket.get('stats', None)
            if stats is not None:
                data['score'] = float(stats.get('score'))
                tags['is_anomaly'] = stats.get('anomaly', False)

            self.insert_times_data(
                index=self.index or index,
                ts=bucket['timestamp'],
                tags=tags,
                data=data,
                timestamp_field=model.timestamp_field,
            )
        self.commit()

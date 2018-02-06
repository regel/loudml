"""
Elasticsearch module for LoudML
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
    Schema,
)

from . import errors
from loudml.datasource import DataSource
from loudml.misc import (
    deepsizeof,
    make_ts,
    parse_addr,
)

# Limit ES aggregations output to 500 MB
PARTITION_MAX_SIZE = 500 * 1024 * 1024

def _date_range_to_ms(from_date=None, to_date=None):
    """
    Convert date range to millisecond
    """
    return (
        None if from_date is None else int(make_ts(from_date) * 1000),
        None if to_date is None else int(make_ts(to_date) * 1000),
    )

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
        'routing': str,
    })

    def __init__(self, cfg):
        cfg['type'] = 'elasticsearch'
        super().__init__(cfg)
        self._es = None

    @property
    def addr(self):
        return self.cfg['addr']

    @property
    def index(self):
        return self.cfg['index']

    @property
    def timeout(self):
        return self.cfg.get('timeout', 30)

    @property
    def es(self):
        if self._es is None:
            addr = parse_addr(self.addr, default_port=9200)
            logging.info('connecting to elasticsearch on %s:%d',
                         addr['host'], addr['port'])
            self._es = Elasticsearch([addr], timeout=self.timeout)

        # urllib3 & elasticsearch modules log exceptions, even if they are
        # caught! Disable this.
        urllib_logger = logging.getLogger('urllib3')
        urllib_logger.setLevel(logging.CRITICAL)
        es_logger = logging.getLogger('elasticsearch')
        es_logger.setLevel(logging.CRITICAL)

        return self._es

    def create_index(self, template):
        """
        Create index and put template
        """

        self.es.indices.create(index=self.index)
        self.es.indices.put_template(
            name='{}-template'.format(self.index),
            body=template,
        )

    def delete_index(self):
        """
        Delete index
        """
        self.es.indices.delete(index=self.index, ignore=404)

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

    def insert_data(self, data, doc_type='generic', doc_id=None):
        """
        Insert entry into the index
        """

        req = {
            '_index': self.index,
            '_type': doc_type,
            '_source': data,
        }

        if doc_id is not None:
            req['_id'] = doc_id

        self.enqueue(req)

    def insert_times_data(
        self,
        ts,
        data,
        doc_type='generic',
        doc_id=None,
        timestamp_field='timestamp',
    ):
        """
        Insert time-indexed entry
        """
        data[timestamp_field] = int(ts * 1000)
        self.insert_data(data, doc_type, doc_id)

    @staticmethod
    def _build_aggs(model):
        """
        Build Elasticsearch aggregations
        """

        aggs = {}

        for feature in model.features:
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
                "field": model.term,
              }
            }
          }
        }

        es_params={}
        if model.routing is not None:
            es_params['routing'] = model.routing

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

        try:
            es_res = self.es.search(
                index=self.index,
                size=0,
                body=body,
                params=es_params,
            )
        except elasticsearch.exceptions.TransportError as exn:
            raise errors.TransportError(str(exn))
        except urllib3.exceptions.HTTPError as exn:
            raise errors.DataSourceError(self.name, str(exn))

        return int(es_res['aggregations']['count']['value'])

    def _compute_quad_partition_size(
        self,
        model,
        from_ms,
        to_ms,
    ):
        cardinality = self.get_field_cardinality(model, from_ms, to_ms)
        nb_time_buckets = math.ceil((to_ms - from_ms) / (6 * 3600 * 100))

        quad = {
            "key_as_string": "2016-12-31T18:00:00.000Z",
            "key": 1483207200000,
            "doc_count": 0,
            "extended_stats": {
                "count": 0,
                "min": 0,
                "max": 0,
                "avg": 0.0,
                "sum": 0.0,
                "sum_of_squares": 0.0,
                "variance": 0.0,
                "std_deviation": 0.0,
                "std_deviation_bounds": {
                    "upper": 0.0,
                    "lower": 0.0,
                }
            }
        }

        return cardinality * deepsizeof(quad) * nb_time_buckets * len(model.features)

    @staticmethod
    def build_quadrant_aggs(model):
        # TODO Generic aggregation not implemented yet
        raise NotImplemented()

    @staticmethod
    def read_quadrant_aggs(key, time_buckets):
        # TODO Generic response handling not implemented yet
        raise NotImplemented()

    @classmethod
    def _build_quadrant_query(
        cls,
        model,
        from_ms=None,
        to_ms=None,
        term_val=None,
        partition=0,
        num_partition=1,
    ):
        body = {
            "size": 0,
            "aggs": {
                "term_val": {
                    "terms": {
                        "field": model.term,
                        "size": model.max_terms,
                        "collect_mode" : "breadth_first",
                        "include": {
                            "partition": partition,
                            "num_partitions": num_partition,
                        },
                    },
                    "aggs": {
                        "values": {
                            "date_histogram": {
                                "field": model.timestamp_field,
                                "interval": "6h",
                                "min_doc_count": 0,
                                "extended_bounds": _build_extended_bounds(from_ms, to_ms - 6 * 3600),
                            },
                            "aggs": cls.build_quadrant_aggs(model),
                        }
                    }
                }
            }
        }

        must = []

        date_range = _build_date_range(model.timestamp_field, from_ms, to_ms)
        if date_range is not None:
            must.append(date_range)

        if term_val is not None:
            must.append({"match": {model.term: term_val}})

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
        from_date=None,
        to_date=None,
        term_val=None,
    ):
        es_params={}
        if model.routing is not None:
            es_params['routing'] = model.routing

        from_ms, to_ms = _date_range_to_ms(from_date, to_date)

        if term_val is None:
            size = self._compute_quad_partition_size(model, from_ms, to_ms)
            num_partition = max(math.ceil(size / PARTITION_MAX_SIZE), 1)
        else:
            num_partition = 1

        for partition in range(0, num_partition):
            logging.info("running aggregations for model '%s', partition %d/%d",
                         model.name, partition, num_partition)

            body = self._build_quadrant_query(
                model,
                from_ms=from_ms,
                to_ms=to_ms,
                term_val=term_val,
                partition=partition,
                num_partition=num_partition,
            )

            try:
                es_res = self.es.search(
                    index=self.index,
                    size=0,
                    body=body,
                    params=es_params,
                )
            except elasticsearch.exceptions.TransportError as exn:
                raise errors.TransportError(str(exn))
            except urllib3.exceptions.HTTPError as exn:
                raise errors.DataSourceError(self.name, str(exn))

            for term_bucket in es_res['aggregations']['term_val']['buckets']:
                yield self.read_quadrant_aggs(term_bucket['key'], term_bucket['values']['buckets'])

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
                "extended_bounds": _build_extended_bounds(from_ms, to_ms),
                "interval": "%ds" % model.bucket_interval,
                "min_doc_count": 0,
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
            if feature.default is np.nan:
                logging.info(
                    "missing data: field '%s', metric: '%s', bucket: %s",
                    feature.field, feature.metric, bucket['key'],
                )
            agg_val = feature.default

        return agg_val

    def get_times_data(
        self,
        model,
        from_date=None,
        to_date=None,
    ):
        features = model.features
        nb_features = len(features)

        es_params={}
        if model.routing is not None:
            es_params['routing'] = model.routing

        from_ms, to_ms = _date_range_to_ms(from_date, to_date)

        body = self._build_times_query(
            model,
            from_ms=from_ms,
            to_ms=to_ms,
        )

        try:
            es_res = self.es.search(
                index=self.index,
                size=0,
                body=body,
                params=es_params,
            )
        except elasticsearch.exceptions.TransportError as exn:
            raise errors.TransportError(str(exn))
        except urllib3.exceptions.HTTPError as exn:
            raise errors.DataSourceError(self.name, str(exn))

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
            X = np.zeros(nb_features, dtype=float)
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

    def save_timeseries_prediction(self, prediction, model):
        for bucket in prediction.format_buckets():
            self.insert_times_data(
                doc_type='prediction_{}'.format(model_name),
                ts=bucket['timestamp'],
                data=bucket['predicted'],
            )
        self.commit()

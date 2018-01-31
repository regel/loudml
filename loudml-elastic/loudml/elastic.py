"""
Elasticsearch module for LoudML
"""

import datetime
import logging

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
    parse_addr,
)

def get_date_range(field, from_date=None, to_date=None):
    """
    Build date range for search query
    """

    date_range = {}

    if from_date is not None:
        date_range['gte'] = from_date
    if to_date is not None:
        date_range['lt'] = to_date

    if len(date_range) == 0:
        return None

    return {'range': {
        field: date_range,
    }}

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

    def insert_times_data(self, ts, data, doc_type='generic', doc_id=None):
        """
        Insert time-indexed entry
        """
        data['timestamp'] = int(ts * 1000)
        self.insert_data(data, doc_type, doc_id)

    def _get_es_agg(
            self,
            model,
            from_date_ms=None,
            to_date_ms=None,
        ):
        body = {
          "size": 0,
          "query": {
            "bool": {
              "must": [
              ],
              "should": [
              ],
              "minimum_should_match": 0,
            }
          },
          "aggs": {
            "histogram": {
              "date_histogram": {
                "field": "timestamp",
                "extended_bounds": {
                    "min": from_date_ms,
                    "max": to_date_ms,
                },
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
        must.append(get_date_range('timestamp', from_date_ms, to_date_ms))
        if len(must) > 0:
            body['query'] = {
                'bool': {
                    'must': must,
                }
            }

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

        body = self._get_es_agg(
            model,
            from_date_ms=int(from_date * 1000),
            to_date_ms=int(to_date * 1000),
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

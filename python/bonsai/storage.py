import argparse
import logging
import time
import os
import json

import elasticsearch.exceptions
import urllib3.exceptions

from elasticsearch import Elasticsearch
from elasticsearch import helpers
from elasticsearch import TransportError

import numpy as np

from . import (
    StorageException,
)

get_current_time = lambda: int(round(time.time()))

def get_date_range(field, from_date=None, to_date=None):
    """
    Build date range for search query
    """

    date_range = {}

    if from_date is not None:
        date_range['gte'] = from_date
    if to_date is not None:
        date_range['lte'] = to_date

    if len(date_range) == 0:
        return None

    return {'range': {
        field: date_range,
    }}

class HTTPError(StorageException):
    """HTTP error"""

class Model:

    def __init__(
            self,
            _id,
            storage,
            index,
            routing,
            name,
            offset,
            span,
            bucket_interval,
            interval,
            features,
            state,
        ):
        self._name = name
        self._id = _id
        self._storage = storage
        self._index = index
        self._routing = routing
        self._offset = offset
        self._span = span
        self._bucket_interval = bucket_interval
        self._interval = interval
        self._features = sorted(features, key=lambda k: k['name'])
        self._state = state
        self.Y_ = None

    def get_es_agg(
            self,
            from_date=None,
            to_date=None,
        ):
        body = {
          "timeout": "10s",
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
                "field": "@timestamp",
                "extended_bounds": {
                    "min": from_date,
                    "max": to_date,
                },
                "interval": "%ds" % self._bucket_interval,
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
        must.append(get_date_range('@timestamp', from_date, to_date))
        if len(must) > 0:
            body['query'] = {
                'bool': {
                    'must': must,
                }
            }

        aggs = {}
        for feature in self._features:
            if feature['metric'] == 'std_deviation' \
                or feature['metric'] == 'variance':
                sub_agg = 'extended_stats'
            else:
                sub_agg = 'stats'
            if 'script' in feature:
                aggs[ feature['name'] ] = {
                      sub_agg : {
                        "script": {
                          "lang": "painless",
                          "inline": feature['script'],
                        }
                      }
                    } 
            elif 'field' in feature:
                aggs[ feature['name'] ] = {
                      sub_agg : {
                        "field": feature['field'],
                      }
                    }

        for x in sorted(aggs):
            body['aggs']['histogram']['aggs'][x] = aggs[x]

        return body


    def get_np_data(
            self,
            from_date=None,
            to_date=None,
        ):

        num_features = len(self._features)
        es_params={}
        if self._routing is not None:
            es_params['routing']=self._routing
       
        body = self.get_es_agg(
                   from_date=from_date,
                   to_date=to_date,
               )

        try:
            es_res = self._storage.es.search(
                index=self._index,
                size=0,
                body=body,
                params=es_params,
            )
        except (
            elasticsearch.exceptions.TransportError,
            urllib3.exceptions.HTTPError,
        ) as exn:
            logging.error("get_np_data: %s", str(exn))
            raise StorageException(str(exn))

        hits = es_res['hits']['total'] 
        if (hits == 0):
            logging.info('Aggregations for model %s: Missing data' % self._name)
            return

        epoch_ms = 1000 * int(get_current_time())
        min_bound_ms = 1000 * int(get_current_time() / self._bucket_interval) * self._bucket_interval

        t0 = 0
        for k in es_res['aggregations']['histogram']['buckets']:
            timestamp=int(k['key'])
            timeval=k['key_as_string']
            X = np.zeros(num_features, dtype=float)
            i = 0
            for feature in self._features:
                name = feature['name']
                metric = feature['metric']
                agg_val = k[name][metric]
                if 'nan_is_zero' in feature:
                    nan_is_zero = feature['nan_is_zero']
                else:
                    nan_is_zero = False

                if (agg_val is None):
                    logging.info('Aggregations(%s) for model %s: Missing data @timestamp: %s' % (name, self._name, timeval))
                    if (nan_is_zero == True):
                        # Write zeros to encode missing data
                        agg_val = 0
                    else:
                        # Use NaN to encode missing data
                        agg_val = np.nan
                X[i] = agg_val
                i += 1

            try:
                # The last interval contains partial data
                if (timestamp == min_bound_ms):
                    R = float(epoch_ms - min_bound_ms) / (1000 * self._bucket_interval)
                    X = R * X + (1-R) * X_prev
            except (NameError):
                # X_prev not defined. No interleaving required.
                pass

            X_prev = X

            if t0 == 0:
                yield 0, X, timeval
                t0=timestamp
            else:
                yield (timestamp-t0)/1000, X, timeval

    def load_model(self):
        import tempfile
        import base64
        import h5py
        # Note: the import were moved here to avoid the speed penalty
        # in code that imports the storage module
        import tensorflow as tf
        import tensorflow.contrib.keras.api.keras.models
        from tensorflow.contrib.keras.api.keras.models import model_from_json

        loss_fct = self._state['loss_fct']
        optimizer = self._state['optimizer']
        mins = np.array(self._state['mins'])
        maxs = np.array(self._state['maxs'])

        loaded_model_json = base64.b64decode(self._state['graph'].encode('utf-8')).decode('utf-8')
        loaded_model = model_from_json(loaded_model_json)

        fd, path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(base64.b64decode(self._state['weights'].encode('utf-8')))
                tmp.close()
        finally:
            #load weights into new model
            loaded_model.load_weights(path)
            os.remove(path)
    
        #compile and evaluate loaded model
        loaded_model.compile(loss=loss_fct, optimizer=optimizer)
        graph = tf.get_default_graph()
    
        return loaded_model, graph, mins, maxs
    
    def save_model(
            self,
            model,
            mins,
            maxs,
            best_params,
        ):
        import tempfile
        import base64
        import h5py

        #Save the model
        # serialize model to JSON
        model_json = base64.b64encode(model.to_json().encode('utf-8'))
        fd, path = tempfile.mkstemp()
        try:
            model.save_weights(path)
            with os.fdopen(fd, 'rb') as tmp:
                serialized = base64.b64encode(tmp.read())
        finally:
            os.remove(path)

        self._storage.save_keras_model(self._id,
                                       model_json.decode('utf-8'),
                                       serialized.decode('utf-8'),
                                       best_params,
                                       mins.tolist(),
                                       maxs.tolist())

    
class Storage:
    def __init__(self, addr, vlan='*'):
        self.addr = addr
        self._es = None
        self._model_index = '.bonsai'
        self._ano_index = '.ml-anomalies-custom-%s' % vlan

    @property
    def es(self):
        if self._es is None:
            addr = self.addr.split(':')
            addr = {
                'host': 'localhost' if len(addr[0]) == 0 else addr[0],
                'port': 9200 if len(addr) == 1 else int(addr[1]),
            }
            logging.info('connecting to elasticsearch on %s:%d',
                         addr['host'], addr['port'])
            self._es = Elasticsearch([addr], timeout=30)

        # urllib3 & elasticsearch modules log exceptions, even if they are
        # caught! Disable this.
        urllib_logger = logging.getLogger('urllib3')
        urllib_logger.setLevel(logging.CRITICAL)
        es_logger = logging.getLogger('elasticsearch')
        es_logger.setLevel(logging.CRITICAL)

        return self._es

    def save_keras_model(
            self,
            _id,
            model_json,
            model_weights,
            best_params,
            mins,
            maxs,
        ):
        es_params={}
        es_params['refresh']='true'
        try:
            doc = { 'doc': { '_state' : {
                'graph': model_json,
                'weights': model_weights, # H5PY data encoded in base64
                'loss_fct': best_params['loss_fct'],
                'optimizer': best_params['optimizer'],
                'best_params': best_params,
                'mins': mins,
                'maxs': maxs,
            }}}

            es_res = self.es.update(
                index=self._model_index,
                id=_id,
                doc_type='model',
                body=doc,
                params=es_params,
            )
        except (
            elasticsearch.exceptions.TransportError,
            urllib3.exceptions.HTTPError,
        ) as exn:
            logging.error("save_keras_model: %s", str(exn))
            raise StorageException(str(exn))

    def create_model(
        self,
        name,
        index,
        routing,
        offset,
        span,
        bucket_interval,
        interval,
        features,
        ):
        es_params={}
        es_params['refresh']='true'
        supported_metrics = [ 
            'sum',
            'count',
            'avg',
            'min',
            'max',
            'std_deviation',
            'variance' ]

        try:
            for feat in features:
                metric = feat['metric'] 
                if not metric in supported_metrics:
                    raise Exception("Unsupported metric: %s" % metric)

            document = {
                'name': name,
                'index': index,
                'routing': routing,
                'offset': offset,
                'span': span,
                'bucket_interval': bucket_interval,
                'interval': interval,
                'features': features,
            }

            es_res = self.es.index(
                index=self._model_index,
                id=None,
                doc_type='model',
                body=document,
                params=es_params,
            )
        except (
            elasticsearch.exceptions.TransportError,
            urllib3.exceptions.HTTPError,
        ) as exn:
            logging.error("create_model: %s", str(exn))
            raise StorageException(str(exn))

    def delete_model(
            self,
            name,
        ):
        try:
            body = {
                'timeout': "10s",
                'query': [
                    {'timestamp': {'order': 'desc'}},
                ],
            }
            must = [
                {'match': {'name': name}},
            ]
            if len(must) > 0:
                body['query'] = {
                    'bool': {
                        'must': must,
                    }
                }

            es_res = self.es.delete_by_query(
                index=self._model_index,
                doc_type='model',
                body=body,
            )
        except (
            elasticsearch.exceptions.TransportError,
            urllib3.exceptions.HTTPError,
        ) as exn:
            logging.error("delete_model: %s", str(exn))
            raise StorageException(str(exn))

    def get_model_list(
            self,
            size=10,
        ):
        try:
            body = {
                'timeout': "10s",
                'size': size,
                'query': {
                    'bool': {
                        'must': [],
                    }
                }
            }

            es_res = self.es.search(
                index=self._model_index,
                doc_type='model',
                body=body,
            )
        except (
            elasticsearch.exceptions.TransportError,
            urllib3.exceptions.HTTPError,
        ) as exn:
            logging.error("get_model_list: %s", str(exn))
            raise StorageException(str(exn))

        res = [ row['_source'] for row in es_res['hits']['hits'] ]
        return res

    def find_model(
            self,
            name,
        ):
        try:
            body = {
                'timeout': "10s",
                'query': [
                    {'timestamp': {'order': 'desc'}},
                ],
            }
            must = [
                {'match': {'name': name}},
            ]
            if len(must) > 0:
                body['query'] = {
                    'bool': {
                        'must': must,
                    }
                }

            es_res = self.es.search(
                index=self._model_index,
                doc_type='model',
                body=body,
            )
        except (
            elasticsearch.exceptions.TransportError,
            urllib3.exceptions.HTTPError,
        ) as exn:
            logging.error("find_model: %s", str(exn))
            raise StorageException(str(exn))

        return es_res['hits']['total'] > 0

    def get_model(
            self,
            name,
        ):
        try:
            body = {
                'timeout': "10s",
                'query': [
                    {'timestamp': {'order': 'desc'}},
                ],
            }
            must = [
                {'match': {'name': name}},
            ]
            if len(must) > 0:
                body['query'] = {
                    'bool': {
                        'must': must,
                    }
                }

            es_res = self.es.search(
                index=self._model_index,
                doc_type='model',
                body=body,
            )
        except (
            elasticsearch.exceptions.TransportError,
            urllib3.exceptions.HTTPError,
        ) as exn:
            logging.error("get_model: %s", str(exn))
            raise StorageException(str(exn))

        _id = es_res['hits']['hits'][0]['_id']
        res = es_res['hits']['hits'][0]['_source']
        if not '_state' in res:
            res['_state'] = None

        return Model(
            storage=self,
            _id=_id,
            index=res['index'],
            routing=res['routing'],
            name=res['name'],
            offset=res['offset'],
            span=res['span'],
            bucket_interval=res['bucket_interval'],
            interval=res['interval'],
            features=res['features'],
            state=res['_state'])



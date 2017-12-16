import argparse
import logging
import time
import os
import json
import uuid

import elasticsearch.exceptions
import urllib3.exceptions

from elasticsearch import Elasticsearch
from elasticsearch import helpers
from elasticsearch import TransportError

import numpy as np
import math

from .util import getsize
from .util import DictDiffer

# limit ES query output to 500 MB
g_max_partition_mem = 500 * 1024 * 1024

def notify(anomaly):
#FIXME: notifications not implemented yet
    return

class StorageException(Exception):
    """Storage exception"""
    def __init__(self, msg="Query to database failed", code=500):
        super().__init__(msg)
        self._code = code

_SUNSHINE_NUM_FEATURES = 4 * 9

def map_quadrant_names(data):
    """
    Build structured quadrant data from a numpy array
    """

    return { 'quadrant_0': {
               '*': {
                 'count': data[0],
                 'duration_avg': data[1],
                 'duration_std': data[2],
               },
               'international': {
                 'count': data[3],
                 'duration_avg': data[4],
                 'duration_std': data[5],
               },
               'premium': {
                 'count': data[6],
                 'duration_avg': data[7],
                 'duration_std': data[8],
               },
             },
             'quadrant_1': {
               '*': {
                 'count': data[9],
                 'duration_avg': data[10],
                 'duration_std': data[11],
               },
               'international': {
                 'count': data[12],
                 'duration_avg': data[13],
                 'duration_std': data[14],
               },
               'premium': {
                 'count': data[15],
                 'duration_avg': data[16],
                 'duration_std': data[17],
               },
             },
             'quadrant_2': {
               '*': {
                 'count': data[18],
                 'duration_avg': data[19],
                 'duration_std': data[20],
               },
               'international': {
                 'count': data[21],
                 'duration_avg': data[22],
                 'duration_std': data[23],
               },
               'premium': {
                 'count': data[24],
                 'duration_avg': data[25],
                 'duration_std': data[26],
               },
             },
             'quadrant_3': {
               '*': {
                 'count': data[27],
                 'duration_avg': data[28],
                 'duration_std': data[29],
               },
               'international': {
                 'count': data[30],
                 'duration_avg': data[31],
                 'duration_std': data[32],
               },
               'premium': {
                 'count': data[33],
                 'duration_avg': data[34],
                 'duration_std': data[35],
               },
             },
           }
 
get_current_time = lambda: int(round(time.time()))

def get_account_cond(field, account_name):
    """
    Build account name filter for search query
    """

    return {'match': {
        field: account_name,
    }}

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

def calc_quadrants(agg):
    num_features = _SUNSHINE_NUM_FEATURES
    profile=np.zeros(num_features)
    for l in agg:
        timestamp=l['key']
        timeval=l['key_as_string']
        s=l['duration_stats']
        _count = float(s['count'])
        if _count == 0:
            continue
        quadrant = int( ((int(timestamp) / (3600*1000)) % 24)/6 )
        _min = float(s['min'])
        _max = float(s['max'])
        _avg = float(s['avg'])
        _sum = float(s['sum'])
        _sum_of_squares = float(s['sum_of_squares'])
        _variance = float(s['variance'])
        _std_deviation = float(s['std_deviation'])
        
        X = np.array( [_count, _sum, _sum_of_squares] )
        profile[(quadrant*9):(quadrant*9 +3)] += X

        s=l['international']['duration_stats']
        _count = s['count']
        if _count != 0:
            _min = float(s['min'])
            _max = float(s['max'])
            _avg = float(s['avg'])
            _sum = float(s['sum'])
            _sum_of_squares = float(s['sum_of_squares'])
            _variance = float(s['variance'])
            _std_deviation = float(s['std_deviation'])

            X = np.array( [_count, _sum, _sum_of_squares] )
            profile[(quadrant*9 +3):(quadrant*9 +6)] += X

        s=l['premium']['duration_stats']
        _count = s['count']
        if _count != 0:
            _min = float(s['min'])
            _max = float(s['max'])
            _avg = float(s['avg'])
            _sum = float(s['sum'])
            _sum_of_squares = float(s['sum_of_squares'])
            _variance = float(s['variance'])
            _std_deviation = float(s['std_deviation'])

            X = np.array( [_count, _sum, _sum_of_squares] )
            profile[(quadrant*9 +6):(quadrant*9 +9)] += X

    for quadrant in range(4):
        for j in range(3):
            _count = profile[quadrant*9 + 3*j]
            _sum = profile[quadrant*9 + 3*j +1]
            _sum_of_squares = profile[quadrant*9 + 3*j +2]
            if _count > 0:
                profile[quadrant*9 + 3*j +1] = _sum / _count
                profile[quadrant*9 + 3*j +2] = math.sqrt(_sum_of_squares/_count - (_sum/_count)**2)

    return profile

class HTTPError(StorageException):
    """HTTP error"""


class NNSOM:

    def __init__(
            self,
            _id,
            storage,
            index,
            routing,
            name,
            offset,
            interval,
            span,
            term,
            max_terms,
            map_w,
            map_h,
            state,
            threshold,
        ):
        self._name = name
        self._id = _id
        self._storage = storage
        self._index = index
        self._routing = routing
        self._offset = offset
        self._interval = interval
        self._span = span
        self._term = term
        self._max_terms = max_terms
        self._map_w = map_w
        self._map_h = map_h
        self._state = state
        self._threshold = threshold
        self._anomalies = {}

    def get_es_agg(
            self,
            from_date=None,
            to_date=None,
            account_name=None,
            partition=0,
            num_partitions=1,
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
            "account": {
              "terms": {
                "field": self._term,
                "size": self._max_terms,
                "collect_mode" : "breadth_first",
                "include": {
                  "partition": partition,
                  "num_partitions": num_partitions,
                },
              },
              "aggs": {
                "count": {
                  "date_histogram": {
                    "field": "@timestamp",
                    "interval": "6h",
                    "min_doc_count": 0,
                    "extended_bounds": {
                      "min": from_date,
                      "max": to_date,
                    }
                  },
                  "aggs": {
                    "duration_stats": {
                      "extended_stats": {
                        "field": "duration"
                      }
                    },
                    "international": {
                      "filter": {
                        "script": {
                        "script": {
                          "lang": "expression",
                          "inline": "doc['international'].value"
                        }
                        }
                      },
                      "aggs": {
                        "duration_stats": {
                          "extended_stats": {
                            "field": "duration"
                          }
                        }
                      }
                    },
                    "premium": {
                      "filter": {
                        "script": {
                        "script": {
                          "lang": "expression",
                          "inline": "doc['toll_call'].value"
                        }
                        }
                      },
                      "aggs": {
                        "duration_stats": {
                          "extended_stats": {
                            "field": "duration"
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }

        must = []
        must.append(get_date_range('@timestamp', from_date, to_date))
        if account_name is not None:
            must.append(get_account_cond(self._term, account_name))

        if len(must) > 0:
            body['query'] = {
                'bool': {
                    'must': must,
                }
            }

        return body

    def get_cardinality(self,
                        from_date=None,
                        to_date=None,
        ):
        body = {
          "size": 0,
          "query": {
            "bool": {
              "must": [
              ],
            }
          },
          "aggs": {
            "count": {
              "cardinality": {
                "field": self._term
              }
            }
          }
        }
        es_params={}
        if self._routing is not None:
            es_params['routing']=self._routing
        
        must = []
        must.append(get_date_range('@timestamp', from_date, to_date))
        if len(must) > 0:
            body['query'] = {
                'bool': {
                    'must': must,
                }
            }

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
            logging.error("get_cardinality: %s", str(exn))
            raise StorageException(str(exn))

        return int(es_res['aggregations']['count']['value'])

    def partition_mem(
            self,
            from_date=None,
            to_date=None,
        ):
        quad = {
                "key_as_string": "2016-12-31T18:00:00.000Z",
                "key": 1483207200000,
                "doc_count": 0,
                "duration_stats": {
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
                    "lower": 0.0
                  }
                },
                "premium": {
                  "doc_count": 0,
                  "duration_stats": {
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
                      "lower": 0.0
                    }
                  }
                },
                "international": {
                  "doc_count": 0,
                  "duration_stats": {
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
                      "lower": 0.0
                    }
                  }
                }
              }
        return self.get_cardinality(from_date, to_date) * \
            getsize(quad) * 4 * ((to_date - from_date) / (24 * 3600 * 1000))

    def get_profile_data(
            self,
            from_date=None,
            to_date=None,
            account_name=None,
        ):
        global g_max_partition_mem
        es_params={}
        if self._routing is not None:
            es_params['routing']=self._routing
        
        if account_name is None:
            mem = self.partition_mem(from_date, to_date)
            num_partitions = math.ceil(mem / g_max_partition_mem)
        else:
            num_partitions = 1

        for partition in range(0,num_partitions):
            logging.info('Running aggregations for model %s partition %d/%d'
                % (self._name, partition, num_partitions))
            body = self.get_es_agg(
                       from_date=from_date,
                       to_date=to_date,
                       account_name=account_name,
                       partition=partition,
                       num_partitions=num_partitions,
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
                logging.error("get_profile_data: %s", str(exn))
                raise StorageException(str(exn))
    
            hits = es_res['hits']['total']
            logging.info('Aggregations for model %s partition %d/%d got terms: %d doc_hits: %d'
                % (self._name, partition, num_partitions,
                   len(es_res['aggregations']['account']['buckets']), hits))
    
            for k in es_res['aggregations']['account']['buckets']:
                account=k['key']
                val=k['count']['buckets']
                profile = calc_quadrants(val)
                yield account, profile


    def set_anomalies(self,
                      anomalies
        ):
        
        d = DictDiffer(anomalies, self._anomalies)
        
        for k in d.changed():
            anomalies[k]['uuid'] = self._anomalies[k]['uuid']
            anomalies[k]['@timestamp'] = self._anomalies[k]['@timestamp']
            anomalies[k]['max_score'] = max(self._anomalies[k]['max_score'],
                                            anomalies[k]['score'])
        for k in d.added():
            anomalies[k]['max_score'] = anomalies[k]['score'] 
            anomalies[k]['uuid'] = str(uuid.uuid4())
            notify(anomalies[k])
        
        self._anomalies = anomalies
        self._storage.index_anomalies(self._anomalies)

    def is_trained(self):
        return (self._state is not None and 'ckpt' in self._state)

    def load_model(self):
        import tempfile
        import base64
        from .som import SOM

        _means = np.array(self._state['means'])
        _stds = np.array(self._state['stds'])
        _model = None
        fd, base_path = tempfile.mkstemp()
        try:
            with open(base_path + '.data-00000-of-00001', 'wb') as tmp:
                tmp.write(base64.b64decode(self._state['ckpt'].encode('utf-8')))
                tmp.close()
            with open(base_path + '.index', 'wb') as tmp:
                tmp.write(base64.b64decode(self._state['index'].encode('utf-8')))
                tmp.close()
            with open(base_path + '.meta', 'wb') as tmp:
                tmp.write(base64.b64decode(self._state['meta'].encode('utf-8')))
                tmp.close()
        except(Exception) as exn:
            logging.error("load_model(): %s", str(exn))
        finally:
            # load weights into new model
            data_dimens = _SUNSHINE_NUM_FEATURES
            _model = SOM(self._map_w, self._map_h, data_dimens, 0)
            _model.restore_model(base_path)
            os.remove(base_path)
            os.remove(base_path + '.data-00000-of-00001')
            os.remove(base_path + '.index')
            os.remove(base_path + '.meta')

        return _model, _means, _stds
    
    def save_model(
            self,
            model,
            means,
            stds,
            mapped_info,
        ):
        import tempfile
        import base64
        from .som import SOM

        # serialize model to base64
        fd, base_path = tempfile.mkstemp()
        try:
            model.save_model(base_path)
            with open(base_path + '.data-00000-of-00001', 'rb') as tmp:
                data = base64.b64encode(tmp.read())
                tmp.close()
            with open(base_path + '.index', 'rb') as tmp:
                idx = base64.b64encode(tmp.read())
                tmp.close()
            with open(base_path + '.meta', 'rb') as tmp:
                meta = base64.b64encode(tmp.read())
                tmp.close()
        except(Exception) as exn:
            logging.error("save_model(): %s", str(exn))
        finally:
            os.remove(base_path)
            os.remove(base_path + '.data-00000-of-00001')
            os.remove(base_path + '.index')
            os.remove(base_path + '.meta')

        s_mapped_info = json.dumps(mapped_info)
        b_mapped_info = base64.b64encode(s_mapped_info.encode('utf-8'))
        self._storage.save_ivoip_model(self._id,
                                       data.decode('utf-8'),
                                       idx.decode('utf-8'),
                                       meta.decode('utf-8'),
                                       b_mapped_info.decode('utf-8'),
                                       means.tolist(),
                                       stds.tolist(),
                                       )

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
            threshold,
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
        self._threshold = threshold

    def get_es_agg(
            self,
            from_date=None,
            to_date=None,
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

    def is_trained(self):
        return (self._state is not None and 'weights' in self._state)

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
    def __init__(self, addr, label=None, timeout=60):
        self.addr = addr
        self.timeout = timeout # global request timeout, if unspecified
        self._es = None
        if label is None:
            self._model_index = '.loudml'
            self._ano_index = '.loudml-anomalies'
        else:
            self._model_index = '.loudml-%s' % label
            self._ano_index = '.loudml-anomalies-%s' % label

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
            self._es = Elasticsearch([addr], timeout=self.timeout)

        # urllib3 & elasticsearch modules log exceptions, even if they are
        # caught! Disable this.
        urllib_logger = logging.getLogger('urllib3')
        urllib_logger.setLevel(logging.CRITICAL)
        es_logger = logging.getLogger('elasticsearch')
        es_logger.setLevel(logging.CRITICAL)

        return self._es

    def index_anomalies(
        self,
        anomalies,
        ):

        data=[]
        for _, doc in anomalies.items():
            data.append({
                '_index': self._ano_index,
                '_type': 'anomaly',
                '_id': doc['uuid'],
                '_source': doc,
            })

        try:
            helpers.bulk(self.es,
                         data)
        except (
            elasticsearch.exceptions.TransportError,
            elasticsearch.helpers.BulkIndexError,
            urllib3.exceptions.HTTPError,
        ) as exn:
            logging.error("index_anomalies: %s", str(exn))
            raise StorageException(str(exn))

    def set_interval(
        self,
        name,
        interval,
        ):
        try:
            body = {
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
                request_timeout=10,
            )
        except (
            elasticsearch.exceptions.TransportError,
            urllib3.exceptions.HTTPError,
        ) as exn:
            logging.error("set_interval: %s", str(exn))
            raise StorageException(str(exn))

        _id = es_res['hits']['hits'][0]['_id']
        es_params={}
        es_params['refresh']='true'
        try:
            doc = { 'doc': { 'interval' : interval
            }}

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
            logging.error("set_interval: %s", str(exn))
            raise StorageException(str(exn))

    def set_threshold(
        self,
        name,
        threshold,
        ):
        try:
            body = {
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
                request_timeout=10,
            )
        except (
            elasticsearch.exceptions.TransportError,
            urllib3.exceptions.HTTPError,
        ) as exn:
            logging.error("set_threshold: %s", str(exn))
            raise StorageException(str(exn))

        _id = es_res['hits']['hits'][0]['_id']
        es_params={}
        es_params['refresh']='true'
        try:
            doc = { 'doc': { 'threshold' : threshold
            }}

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
            logging.error("set_threshold: %s", str(exn))
            raise StorageException(str(exn))

    def save_ivoip_model(
        self,
        _id,
        model_ckpt,
        model_idx,
        model_meta,
        mapped_info,
        means,
        stds,
        ):
        es_params={}
        es_params['refresh']='true'
        try:
            doc = { 'doc': { '_state' : {
                        'ckpt': model_ckpt, # TF CKPT data encoded in base64
                        'index': model_idx,
                        'meta': model_meta,
                        'mapped_info': mapped_info,
                        'means': means,
                        'stds': stds,
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
            logging.error("save_ivoip_model: %s", str(exn))
            raise StorageException(str(exn))

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
        threshold=30,
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
                'threshold': threshold,
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

    def create_ivoip(
        self,
        name,
        index,
        routing,
        offset,
        term,
        max_terms,
        interval,
        span,
        map_w,
        map_h,
        threshold=30,
        ):
        es_params={}
        es_params['refresh']='true'

        try:
            document = {
                'name': name,
                'index': index,
                'routing': routing,
                'offset': offset,
                'interval': interval,
                'span': span,
                'term': term,
                'max_terms': max_terms,
                'map_w': map_w,
                'map_h': map_h,
                'threshold': threshold,
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
            logging.error("create_ivoip: %s", str(exn))
            raise StorageException(str(exn))

    def delete_model(
            self,
            name,
        ):
        try:
            body = {
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
                request_timeout=10,
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
                request_timeout=10,
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
                request_timeout=10,
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
                request_timeout=10,
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
            threshold=res['threshold'],
            state=res['_state'])

    def get_ivoip(
            self,
            name,
        ):
        try:
            body = {
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
                request_timeout=10,
            )
        except (
            elasticsearch.exceptions.TransportError,
            urllib3.exceptions.HTTPError,
        ) as exn:
            logging.error("get_ivoip: %s", str(exn))
            raise StorageException(str(exn))

        _id = es_res['hits']['hits'][0]['_id']
        res = es_res['hits']['hits'][0]['_source']
        if not '_state' in res:
            res['_state'] = None

        return NNSOM(
            storage=self,
            _id=_id,
            index=res['index'],
            routing=res['routing'],
            name=res['name'],
            offset=res['offset'],
            interval=res['interval'],
            span=res['span'],
            term=res['term'],
            max_terms=res['max_terms'],
            map_w=res['map_w'],
            map_h=res['map_h'],
            threshold=res['threshold'],
            state=res['_state'],
            )



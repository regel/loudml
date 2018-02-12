"""
iVOIP for LoudML
"""

import logging
import math

import elasticsearch.exceptions
import urllib3.exceptions

import numpy as np

logging.getLogger('tensorflow').disabled = True

from . import errors

from loudml.elastic import (
    ElasticsearchDataSource,
)
from loudml.fingerprints import (
    FingerprintsModel,
)
from loudml.misc import (
    deepsizeof,
    make_ts,
    parse_addr,
)

_SUNSHINE_NUM_FEATURES = 4 * 9

class IVoipDataSource(ElasticsearchDataSource):
    """
    iVOIP datasource

    """

    TYPE = 'ivoip'

    @staticmethod
    def build_quadrant_aggs(self):
        return {
          "duration_stats": {
            "extended_stats": {"field": "duration"}
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
                "extended_stats": {"field": "duration"}
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
                "extended_stats": {"field": "duration"}
              }
            }
          }
        }

    @staticmethod
    def read_quadrant_aggs(key, time_buckets):
        # XXX The buckets are currently returned in Elasticsearch format and read in IVoipModel
        return key, time_buckets


class IVoipFingerprintsModel(FingerprintsModel):
    """
    iVOIP fingerprintsModel

    This model is compatible only with IVoipDataSource
    """

    TYPE = 'ivoip_fingerprints'

    SCHEMA = FingerprintsModel.SCHEMA.extend({
        'features': None, # iVOIP features handling is hard-coded
    })

    _FEATURE_NAMES = [
        "count(total)", "avg(duration)", "std(duration)",
        "count(international)", "avg(international.duration)", "std(international.duration)",
        "count(premium)", "avg(premium.duration)", "std(premium.duration)",
    ]

    _SUNSHINE_NUM_FEATURES = len(_FEATURE_NAMES)

    def __init__(self, settings, state=None):
        super().__init__(settings, state)
        self.timestamp_field = self.settings.get('timestamp_field', '@timestamp')

    @property
    def nb_features(self):
        return self._SUNSHINE_NUM_FEATURES

    @property
    def feature_names(self):
        return self._FEATURE_NAMES

    def format_quadrants(self, agg):
        res = np.zeros(4 * self.nb_features)

        for l in agg:
            timestamp = l['key']
            timeval = l['key_as_string']
            s = l['duration_stats']
            _count = float(s['count'])

            if _count == 0:
                continue

            quadrant = int(((int(timestamp) / (3600 * 1000)) % 24) / 6)
            _min = float(s['min'])
            _max = float(s['max'])
            _avg = float(s['avg'])
            _sum = float(s['sum'])
            _sum_of_squares = float(s['sum_of_squares'])
            _variance = float(s['variance'])
            _std_deviation = float(s['std_deviation'])

            X = np.array([_count, _sum, _sum_of_squares])
            quad_idx = quadrant * 9
            res[quad_idx:quad_idx + 3] += X

            s = l['international']['duration_stats']
            _count = s['count']
            if _count != 0:
                _min = float(s['min'])
                _max = float(s['max'])
                _avg = float(s['avg'])
                _sum = float(s['sum'])
                _sum_of_squares = float(s['sum_of_squares'])
                _variance = float(s['variance'])
                _std_deviation = float(s['std_deviation'])

                X = np.array([_count, _sum, _sum_of_squares])
                res[quad_idx + 3:quad_idx + 6] += X

            s = l['premium']['duration_stats']
            _count = s['count']
            if _count != 0:
                _min = float(s['min'])
                _max = float(s['max'])
                _avg = float(s['avg'])
                _sum = float(s['sum'])
                _sum_of_squares = float(s['sum_of_squares'])
                _variance = float(s['variance'])
                _std_deviation = float(s['std_deviation'])

                X = np.array([_count, _sum, _sum_of_squares])
                res[quad_idx + 6:quad_idx + 9] += X

        for quadrant in range(4):
            for j in range(3):
                quad_idx = quadrant * 9
                _count = res[quad_idx + 3 * j]
                _sum = res[quad_idx + 3 * j + 1]
                _sum_of_squares = res[quad_idx + 3 * j + 2]

                if _count > 0:
                    mean = _sum / _count
                    variance = math.sqrt(_sum_of_squares / _count - (_sum/_count) ** 2)
                    res[quad_idx + 3 * j + 1] = _sum / _count
                    res[quad_idx + 3 * j + 2] = variance

        return res

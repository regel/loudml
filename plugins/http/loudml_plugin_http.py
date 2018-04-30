# This file is part of LoudML HTTP plug-in. LoudML HTTP plug-in is free software:
# you can redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Red Mint Network

import logging
import requests

from loudml.api import Hook

from voluptuous import (
    All,
    Any,
    Invalid,
    Required,
    Schema,
    Url,
)

CONFIG_SCHEMA = Schema({
    Required('url'): Url(),
    'method': Any('POST', 'PUT', 'GET'),
})

class HTTPHook(Hook):
    @staticmethod
    def validate(config):
        try:
            CONFIG_SCHEMA(config)
        except Invalid as exn:
            raise ValueError(exn.error_message)

    def on_anomaly(self, model, timestamp, score, predicted, observed, **kwargs):
        try:
            requests.post(self.config['url'], timeout=1, json={
              'model': model,
              'timestamp': timestamp,
              'score': score,
              'predicted': predicted,
              'observed': observed,
            })
        except requests.exceptions.RequestException as exn:
            logging.error("cannot notify %s: %s", self.config['url'], exn)

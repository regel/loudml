# This file is part of LoudML Plug-In API. LoudML Plug-In API is free software:
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

"""
LoudML public API
"""

import os
import yaml

from voluptuous import (
    ALLOW_EXTRA,
    Any,
    Schema,
)

def validate(schema, data):
    """
    Validate data against a schema
    """
    return schema(data) if schema else data

class Plugin:
    """
    Base class for LoudML plug-in
    """

    CONFIG_SCHEMA = None
    instance = None

    def __init__(self, name, config_dir):
        self.set_instance(self)
        self.name = name
        self.config = None
        path = os.path.join(config_dir, 'plugins.d', '{}.yml'.format(name))

        try:
            with open(path) as config_file:
                config = yaml.load(config_file)
        except FileNotFoundError:
            # XXX No config or plug-in disabled
            return

        self.config = self.validate(config)

    @classmethod
    def validate(cls, config):
        """
        Validate plug-in configuration
        """
        return validate(cls.CONFIG_SCHEMA, config)

    @classmethod
    def set_instance(cls, instance):
        cls.instance = instance


class Hook:
    """
    Generate notification
    """
    CONFIG_SCHEMA = None

    def __init__(self, name, config=None):
        self.name = name
        self.config = self.validate(config)

    @classmethod
    def validate(cls, config):
        """
        Validate hook configuration
        """
        return validate(cls.CONFIG_SCHEMA, config)

    def on_anomaly_start(
        self,
        model,
        dt,
        score,
        predicted,
        observed,
        *args,
        **kwargs
    ):
        """
        Callback function called on anomaly detection

        model -- model name
        dt -- Timestamp of the anomaly as a datetime.datetime object
        score -- Computed anomaly score [0-100]
        predicted -- Predicted values
        observed -- Observed values
        mse -- MSE
        dist -- Distance
        """
        raise NotImplemented()

    def on_anomaly_end(self, model, dt, score, *args, **kwargs):
        """
        Callback function called on anomaly detection

        model -- model name
        dt -- Timestamp of the anomaly as a datetime.datetime object
        score -- Computed anomaly score [0-100]
        """
        pass

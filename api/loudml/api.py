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

class Hook:
    """
    Generate notification
    """

    def __init__(self, name, config=None):
        self.name = name
        self.config = config

    def on_anomaly(self, model, timestamp, score, predicted, observed, **kwargs):
        """
        Callback function called on anomaly detection

        timestamp -- UNIX timestamp of the anomaly
        score -- Computed anomaly score [0-100]
        predicted -- Predicted values
        observed -- Observed values
        mse -- MSE
        dist -- Distance
        """
        raise NotImplemented()

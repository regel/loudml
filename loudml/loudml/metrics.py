"""
Collect and send metrics about program usage
"""

import pkg_resources
import requests

from loudml.license import License


def send_metrics():
    url = 'http://telemetry.loudml.io/api'
    data = {
        'host_id': License.my_host_id(),
        'loudml': {
            'version': pkg_resources.get_distribution("loudml").version,
        },
    }
    headers = {
        'user-agent': 'loudml',
    }

    requests.post(url, json=data, headers=headers)

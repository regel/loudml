"""
Collect and send metrics about program usage
"""

import configparser
import io
import pkg_resources
import requests

from loudml.license import License


# Workaround for ConfigParser requiring sections
# https://mail.python.org/pipermail/python-dev/2002-November/029987.html
class MyConfigParser(configparser.ConfigParser):
    def read(self, filename):
        try:
            text = open(filename).read()
        except IOError:
            pass
        else:
            file = io.StringIO("[os-release]\n" + text)
            self.readfp(file, filename)


def send_metrics(config, storage, user_agent="loudmld"):
    """
    Send usage information to telemetry server

     :param config:      telemetry configuration
     :type  config:      dict
     :param storage:     storage backend
     :type  storage:     loudml.Storage
     :param user_agent:  HTTP request user agent
     :type  user_agent:  str
    """
    if not config['enabled']:
        return

    os_release = MyConfigParser()
    os_release.read("/etc/os-release")

    url = 'http://telemetry.loudml.io/api'
    data = {
        'host_id': License.my_host_id(),
        'loudml': {
            'distribution': os_release.get("os-release", "NAME"),
            'nr_models': len(storage.list_models()),
            'version': pkg_resources.get_distribution("loudml").version,
        },
    }
    headers = {
        'user-agent': user_agent,
    }

    try:
        requests.post(url, json=data, headers=headers)
    except Exception:
        # Ignore error as it may be expected in some environments (offline...)
        pass

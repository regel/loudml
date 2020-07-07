"""
Collect and send metrics about program usage
"""

from configparser import (
    ConfigParser, NoSectionError
)
import io
import pkg_resources
import requests

from loudml.misc import my_host_id


# Workaround for ConfigParser requiring sections
# https://mail.python.org/pipermail/python-dev/2002-November/029987.html
class MyConfigParser(ConfigParser):
    def read(self, filename):
        try:
            text = open(filename).read()
        except IOError:
            pass
        else:
            file = io.StringIO("[os-release]\n" + text)
            self.readfp(file, filename)

    def safe_get(self, section, val):
        # Issue #208: /etc/os-release not always present
        # On macOS it is: /System/Library/CoreServices/SystemVersion.plist
        # TODO: consider more granular OS release extraction
        try:
            return self.get(section, val)
        except NoSectionError:
            return "N/A"


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
    if not config['enable']:
        return

    os_release = MyConfigParser()
    os_release.read("/etc/os-release")

    url = 'http://telemetry.loudml.io/api'
    data = {
        'host_id': my_host_id(),
        'loudml': {
            'distribution': os_release.safe_get("os-release", "NAME"),
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

import datetime
import json
import gzip
import hashlib
import logging

from base64 import b64encode, b64decode

from uuid import getnode

from Crypto.PublicKey import RSA
from Crypto.Hash import SHA256
from Crypto.Signature import PKCS1_v1_5

from loudml.misc import make_ts

# License format version (see class License for format details)
LICENSE_VERSION = 1

# Current Loud ML major version
# TODO integrate with setuptools
LOUDML_MAJOR_VERSION = 1

MAX_RUNNING_MODELS = 3

# Use Community Edition restrictions as default
DEFAULT_PAYLOAD = {
    'features': {
        'datasources': ["elasticsearch", "influxdb"],
        'models': ["timeseries"],
        'nrmodels': 3,
    },
    'hostid': "any",
}


def compute_digest(data, certificate, key):
    """
    Compute signature using data and private key

    :param data: data
    :type  data: bytes-like object

    :param key: private key
    :type  key: bytes-like object

    :param certificate: certificate
    :type  certificate: bytes-like object

    :return: signature
    """
    rsakey = RSA.importKey(key)
    h = SHA256.new(data)
    h.update(certificate)

    return PKCS1_v1_5.new(rsakey).sign(h)


class License:
    """
    The license format actually uses two nested formats. The outer format is
    used to verify the license integrity. It is a gzipped set of lines. The
    inner format is a JSON dictionnary. The main reason why everything is not
    in JSON format is to make sure the payload is interpreted as binary data
    to compute digest.

    ```
    VERSION
    PUBLIC_KEY
    CERTIFICATE
    DIGEST
    PAYLOAD
    ```

    `VERSION` is an integer that describes the license format. The rest of
    this documentation applies to version `1`.

    `PUBLIC_KEY` is the public key attached to a customer. The private key is
    not distributed.

    `CERTIFICATE` is a chain of certificates to certify the license has been
    issued by an authorized entity. Currently this is not checked.

    `DIGEST` is a digest of the subsequent payload, computed using the
    aforementioned private key. Any change to the payload or the public key
    will result in a digest validation failure.

    `PAYLOAD` is a JSON structure that describes the actual limitations.

    ```
    {
      "features": {
        "datasources": [ ... ],
        "data_range": [ start, end ],
        "models": [ ... ],
        "nrmodels": ...
      },
      "version": ...,
      "exp_date": ...,
      "hostid": ...,
      "serial_num": ...
    }
    ```

    `features` is a dictionnary of limitations that are checked during the
    program lifetime. In contrast, the other fields at the root of the payload
    structure are checked only when the program starts.

    `features:datasources` is the list of the datasources that are allowed.
    The name of the datasources are the same as the entry points in the files
    `loudml-<datasource>/setup.py`.

    `features:data_range` is a list with the start date and stop date of the
    data that is allowed to be analyzed.

    `features:models` is the list of the models that are allowed. The name of
    the models are the same as the entry points `loudml.models` in `setup.py`.

    `features:nrmodels` is the maximum number of models that can be stored.
    The special value 'unlimited' disables the number check.

    `version` is the major version number for Loud ML. It is defined in this
    file.

    `exp_date` is the expiration date in `YYYY-MM-DD` format.

    `hostid` restricts the hosts on which the license if valid.

    `serial_num` is a serial number attached to the license during generation.
    """

    version = LICENSE_VERSION
    private_key = None
    public_key = None
    certificate = b''
    payload = DEFAULT_PAYLOAD
    payload_raw = None
    digest = None

    def load(self, path):
        """
        Load license from file

        :param path: path to file
        :type  path: str
        """

        with gzip.open(path, 'r') as f:
            lines = f.readlines()

        self.version = int(lines[0])
        self.public_key = b64decode(lines[1])
        self.certificate = b64decode(lines[2])
        self.digest = b64decode(lines[3])
        self.payload_raw = b64decode(lines[4])
        self.payload = json.loads(self.payload_raw.decode('ascii'))

    def save(self, path):
        """
        Save license to file

        :param path: path to file
        :type  path: str
        """
        self.digest = compute_digest(self.payload_raw, self.certificate,
                                     self.private_key)

        lines = "{0}\n{1}\n{2}\n{3}\n{4}".format(
            self.version,
            b64encode(self.public_key).decode('ascii'),
            b64encode(self.certificate).decode('ascii'),
            b64encode(self.digest).decode('ascii'),
            b64encode(self.payload_raw).decode('ascii'))

        with gzip.open(path, 'wt') as f:
            f.write(lines)

    def validate(self):
        """
        Validate license

        :return bool: whether signature is authentic
        """
        h = SHA256.new(self.payload_raw)
        h.update(self.certificate)
        rsakey = RSA.importKey(self.public_key)

        return PKCS1_v1_5.new(rsakey).verify(h, self.digest)

    def has_expired(self):
        """
        Check whether system clock is beyond license expiration date.

        If expiration date is not present, it is interpreted as being a
        perpetual license that does not expire.
        """
        date = self.payload.get('exp_date', None)
        if date is None:
            return False

        exp_date = datetime.datetime.strptime(date, "%Y-%m-%d")

        return datetime.datetime.now() > exp_date

    def version_allowed(self):
        """
        Check whether major version number is allowed.

        Older major versions are also allowed.
        """
        version = self.payload.get('version', LOUDML_MAJOR_VERSION)

        return LOUDML_MAJOR_VERSION <= version

    def host_allowed(self):
        """
        Check whether current host is allowed to run the software.
        """
        host_id = self.payload.get('hostid', 'any')

        if host_id == 'any' or host_id == self.my_host_id():
            return True

    def data_range_allowed(self, from_date, to_date):
        """
        Check whether data range is allowed.

        If data_range is not present, any date is considered valid.
        """
        features = self.payload.get('features', None)
        data_range = features.get('data_range', None)
        if data_range is None:
            return True

        allowed_start = make_ts(data_range[0])
        allowed_end = make_ts(data_range[1])
        check_start = make_ts(from_date)
        check_end = make_ts(to_date)

        return check_start >= allowed_start and check_end <= allowed_end

    @property
    def max_running_models(self):
        features = self.payload.get('features')
        if features:
            return features.get('nrmodels', MAX_RUNNING_MODELS)
        return MAX_RUNNING_MODELS

    @staticmethod
    def my_host_id():
        """
        Compute host identifier.

        Identifier is based on:
        - identifier computed by Python uuid library (usually MAC address)
        - MD5 hashing (to make computation less obvious)

        It is NOT based on:
        - system UUID in DMI entries (requires root privileges and may not be
          avalaible)
        - root filesystem UUID (requires root privileges)

        Obviously, changing the algorithm breaks all the current licenses.
        Consequently changes will require issuing licences for every customer.
        """

        m = hashlib.md5()
        m.update(str(getnode()).encode('ascii'))

        return m.hexdigest()

    def global_check(self):
        """
        Check license status.
        """

        if self.payload == DEFAULT_PAYLOAD:
            return

        if not self.validate():
            raise Exception("license integrity check failure")

        if self.has_expired():
            logging.warning("license expired since " +
                            self.payload['exp_date'])

        if not self.version_allowed():
            raise Exception("software version not allowed")

        if not self.host_allowed():
            raise Exception("host_id not allowed")

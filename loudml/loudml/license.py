import datetime
import json
import gzip
import logging

from base64 import b64encode, b64decode

from Crypto.PublicKey import RSA
from Crypto.Hash import SHA256
from Crypto.Signature import PKCS1_v1_5

# License format version (see class License for format details)
LICENSE_VERSION = 1

# Current LoudML major version
# TODO integrate with setuptools
LOUDML_MAJOR_VERSION=1

# Use Community Edition restrictions as default
DEFAULT_PAYLOAD = {
    'features': {
        'datasources': [ "elasticsearch", "influxdb" ],
        'models': [ "timeseries" ],
        'nrmodels': 3,
    },
    'hostid': "any",
}

def compute_digest(data, certificate, key):
    """
    Compute signature using data and private key

    :param data:
        Data

    :param key:
        Private key

    :param certificate:
        Certificate

    :return:
        Signature
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

    `features:models` is the list of the models that are allowed. The name of
    the models are the same as the entry points `loudml.models` in `setup.py`.

    `features:nrmodels` is the maximum number of models that can be stored.

    `version` is the major version number for LoudML. It is defined in this
    file.

    `exp_date` is the expiration date in `YYYY-MM-DD` format.

    `hostid` restricts the hosts on which the license if valid.

    `serial_num` is a serial number attached to the license during generation.
    """

    version = LICENSE_VERSION
    private_key = None
    public_key = None
    certificate = b''
    payload_raw = None
    digest = None

    @staticmethod
    def default_payload():
        return DEFAULT_PAYLOAD


    def load(self, path):
        """
        Load license from file

        :param path:
            Path to file
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

        :param path:
            Path to file
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

        :return bool:
            Whether signature is authentic
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


    @staticmethod
    def my_host_id():
        """
        Compute host identifier.

        On physical devices, it is the MAC address of the first network
        device. On virtual machines, it is the UUID of the root storage
        device.
        """

        # TODO: implement computation
        return "TBD"


    def global_check(self):
        """
        Check license status.
        """

        if not self.validate():
            raise Exception("license integrity check failure")

        if self.has_expired():
            logging.warning("license expired since " + self.payload['exp_date'])

        if not self.version_allowed():
            raise Exception("software version not allowed")

        if not self.host_allowed():
            raise Exception("host_id not allowed")

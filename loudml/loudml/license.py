from base64 import b64encode, b64decode
import gzip

from Crypto.PublicKey import RSA
from Crypto.Hash import SHA256
from Crypto.Signature import PKCS1_v1_5

LICENSE_VERSION = 1

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
    License
    """
    version = LICENSE_VERSION
    private_key = None
    public_key = None
    certificate = b''
    data = None
    digest = None

    def save(self, path):
        """
        Save license to file

        :param path:
            Path to file
        """
        self.digest = compute_digest(self.data, self.certificate,
                                     self.private_key)

        lines = "{0}\n{1}\n{2}\n{3}\n{4}".format(
            self.version,
            b64encode(self.public_key).decode('ascii'),
            b64encode(self.certificate).decode('ascii'),
            b64encode(self.digest).decode('ascii'),
            b64encode(self.data).decode('ascii'))

        with gzip.open(path, 'wt') as f:
            f.write(lines)


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
        self.data = b64decode(lines[4])


    def validate(self):
        """
        Validate license

        :return bool:
            Whether signature is authentic
        """
        h = SHA256.new(self.data)
        h.update(self.certificate)
        rsakey = RSA.importKey(self.public_key)

        return PKCS1_v1_5.new(rsakey).verify(h, self.digest)

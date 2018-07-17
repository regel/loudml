import datetime
import json

from loudml.license_base import License as LicenseBase

# Current LoudML major version
# TODO integrate with setuptools
LOUDML_MAJOR_VERSION=1

class License(LicenseBase):
    """
    Handle the 'payload' field in the license. It contains details about the
    features that are not described in the general license format.
    """

    def load(self, path):
        super().load(path)
        self.limits = json.loads(self.data.decode('ascii'))
        self.serial_num = self.limits.get('serial_num', 0)

    def has_expired(self):
        """
        Check whether system clock is beyond license expiration date.

        If expiration date is not present, it is interpreted as being a
        perpetual license that does not expire.
        """
        date = self.limits.get('exp_date', None)
        if date is None:
            return False

        exp_date = datetime.datetime.strptime(date, "%Y-%m-%d")

        return datetime.datetime.now() > exp_date

    def version_allowed(self):
        """
        Check whether major version number is allowed.

        Older major versions are also allowed.
        """
        version = self.limits.get('version', LOUDML_MAJOR_VERSION)

        return LOUDML_MAJOR_VERSION >= version

    def host_allowed(self):
        """
        Check whether current host is allowed to run the software.
        """
        host_id = self.limits.get('hostid', 'any')

        if host_id == 'any' or host_id == self.my_host_id():
            return True

    def my_host_id(self):
        """
        Compute host identifier.

        On physical devices, it is the MAC address of the first network
        device. On virtual machines, it is the UUID of the root storage
        device.
        """

        # TODO: implement computation
        return "TBD"

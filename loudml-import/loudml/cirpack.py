import pkg_resources

import ipaddress
from datetime import datetime
import dateutil.parser
import logging

from rmn_common.data_import import Parser
from .phone_rates import PhoneRates


def international_nature(nature):
    return (nature == 4 or nature == 117)


class CdrParser(Parser):
    def __init__(self):
        super().__init__()
        self._phonelib = PhoneRates(local_region='FR')

    def get_template(self, db_name, measurement):
        resource = pkg_resources.resource_filename(__name__, 'resources/phonedb.template')
        content = open(resource, 'rU').read()
        return content.format(db_name, measurement)

    def decode(self, row):
        cols = row.decode('utf-8').split()
        account = cols[2]
        #‘1’ = for an incoming call, ‘0’ = for an outgoing or a transited call.
        direction = int(cols[3])
        ts = dateutil.parser.parse(cols[4] + cols[5], ignoretz=True)
        total_call_duration = int(cols[8])
        # (hexadecimal) IP address of the switch generating the CDR
        ip_addr = ipaddress.ip_address(bytes.fromhex(cols[9]))

# Number Nature
#Undefined 0
#Subscriber number (national use) 1
#Unknown 2
#National number 3
#International number 4
#Network-specific number (national use) 5
#Interworking 8
#Closed user group nature 11
#Truncated number 12
#Special 115 
#Indirect national 116
#Indirect international 117
        calling_number_nature = int(cols[14])
        calling_number = cols[15]
        calling_number_international = international_nature(calling_number_nature)
        if calling_number_international:
            calling_number = "+" + calling_number

        called_number_nature = int(cols[20])
        called_number = cols[21]
        called_number_international = international_nature(called_number_nature)
        if called_number_international:
            called_number = "+" + called_number
        
        stats = self._phonelib.get_stats2(calling_number, called_number)
        tag_dict = {
            'account': account,
        }
        stats.update({
            'direction': direction,
            'duration': total_call_duration,
            'cost': float(stats['rate']) * total_call_duration/60,
        })
        return int(ts.timestamp()), tag_dict, stats

    def read_csv(self, fp, encoding):
        for row in fp:
            yield self.decode(row)



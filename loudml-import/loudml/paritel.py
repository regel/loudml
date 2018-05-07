import pkg_resources

import dateutil.parser
import logging
import pandas as pd

import phonenumbers
from phonenumbers import geocoder
from phonenumbers import PhoneNumberType, PhoneNumberFormat, NumberParseException

from rmn_common.data_import import Parser
from .phone_rates import PhoneRates

phonenumbers.PhoneMetadata.load_all()

def _parse_number(num, local_region, rates):
    y = phonenumbers.parse(num, local_region)

    #if phonenumbers.is_possible_number(y) == False:
    #    print("Isn't possible number:", num)
    #if phonenumbers.is_valid_number(y) == False:
    #    print("Isn't valid number:", num)

    output_num = phonenumbers.format_number(
        y,
        phonenumbers.PhoneNumberFormat.E164,
    )
    region_code = geocoder.region_codes_for_country_code(y.country_code)[0]
    international = False
    mobile = False
    premium = False

    if PhoneNumberType.MOBILE == phonenumbers.number_type(y):
        mobile = True
    if PhoneNumberType.PREMIUM_RATE == phonenumbers.number_type(y):
        premium = True
    if region_code != local_region:
        international = True

    fraud_level, pricing = rates.get_fraud_level_and_rate(output_num[1:])

    return {
        'premium': premium,
        'mobile': mobile,
        'international': international,
        'phonenumber': output_num,
        'region': region_code,
        'fraud_level': fraud_level,
        'pricing': pricing,
    }


def parse_number(num, local_region, rates):
    try:
        return _parse_number(num, local_region, rates)
    except phonenumbers.phonenumberutil.NumberParseException as exn:
        logging.error("invalid number: %s", num)
        return {
            'premium': False,
            'mobile': False,
            'international': False,
            'phonenumber': num,
            'region': 'INVALID',
            'fraud_level': 'A',
            'pricing': 1,
        }

class CdrParser(Parser):
    def __init__(self):
        super().__init__()
        self._rates = PhoneRates()

    def get_template(self, db_name, measurement):
        resource = pkg_resources.resource_filename(__name__, 'resources/phonedb.template')
        content = open(resource, 'rU').read()
        return content.format(db_name, measurement)

    def decode(self, row):
        try:
            ts = dateutil.parser.parse(row['call_start_date'])
        except (ValueError, TypeError):
            logging.error("got invalid 'call_start_date': '%s'", 0)
            print(row)
            raise ValueError

        total_call_duration = int(row['call_duration'])
        calling_dict = parse_number(row['calling_number'], 'FR', self._rates)
        called_dict = parse_number(row['called_number'], 'FR', self._rates)
        calling_number = calling_dict['phonenumber']
        called_number = called_dict['phonenumber']
        calling_number_international = calling_dict['international']
        calling_number_region = calling_dict['region']
        called_number_international = called_dict['international']
        called_number_region = called_dict['region']

        fraud_level = called_dict['fraud_level']
        pricing = called_dict['pricing']

        tag_dict = {
            'account': row['account_ref'],
        }
        row_data = {
            'calling_number': calling_number,
            'calling_number_region': calling_number_region,
            'called_number': called_number,
            'called_number_region': called_number_region,
            'duration': total_call_duration,
            'fraud_level': fraud_level,
            'pseudo_price': pricing * total_call_duration/60,
            'mobile': called_dict['mobile'],
            'international': called_dict['international'],
            'toll_call': called_dict['premium'],
        }
        return int(ts.timestamp()), tag_dict, row_data

    def read_csv(self, fp, encoding):
        df = pd.read_csv(
            fp,
            encoding=encoding,
            delimiter='\t',
            # XXX required because operator column may contain '\r'
            lineterminator='\n',
            dtype={
                'calling_number': str,
                'called_number': str,
                'call_duration': str,
                'call_start_date': str,
                'account_ref': str,
                'operator': str,
            },
        )
        total = len(df)

        for index, row in df.iterrows():
            try:
                nb_read = index + 1
                self.show_progress(nb_read, nb_read / total)
                yield self.decode(row)
            except ValueError:
                logging.error("invalid row %d", index)
                continue

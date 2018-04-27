import pkg_resources

import dateutil.parser
import logging
import pandas as pd

import phonenumbers
from phonenumbers import geocoder
from phonenumbers import PhoneNumberType, PhoneNumberFormat, NumberParseException

from .parser import Parser

phonenumbers.PhoneMetadata.load_all()

def _parse_number(num, local_region):
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

    return {
        'premium': premium,
        'mobile': mobile,
        'international': international,
        'phonenumber': output_num,
        'region': region_code,
    }


def parse_number(num, local_region):
    try:
        return _parse_number(num, local_region)
    except phonenumbers.phonenumberutil.NumberParseException as exn:
        logging.error("invalid number: %s", num)
        return {
            'premium': False,
            'mobile': False,
            'international': False,
            'phonenumber': num,
            'region': 'INVALID',
        }

class CdrParser(Parser):
    def get_template(self, db_name, measurement):
        resource = pkg_resources.resource_filename(__name__, 'resources/paritel.template')
        content = open(resource, 'rU').read()
        return content.format(db_name, measurement)

    def decode(self, row):
        try:
            ts = dateutil.parser.parse(row['call_start_date'])
        except (ValueError, TypeError):
            logging.error("got invalid 'call_start_date': '%s'", 0)
            print(row)
            raise ValueError

        caller = parse_number(row['calling_number'], 'FR')
        callee = parse_number(row['called_number'], 'FR')

        tag_dict = {
            'account': row['account_ref'],
        }
        row_data = {
            'calling_number': caller['phonenumber'],
            'called_number': callee['phonenumber'],
            'duration': int(row['call_duration']),
            'mobile': callee['mobile'],
            'international': callee['international'],
            'toll_call': callee['premium'],
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

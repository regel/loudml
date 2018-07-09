
import loudml.vendor
from loudml.knn import get_groups

import pkg_resources
import csv
from operator import itemgetter
import datetime
import dateutil.parser
import logging

# pycountry==18.5.26
import pycountry
# phonenumbers==8.9.7
import phonenumbers
from phonenumbers import geocoder
from phonenumbers import PhoneNumberType, PhoneNumberFormat, NumberParseException

phonenumbers.PhoneMetadata.load_all()

# biopython 1.71
from Bio import trie
from Bio.triefind import match

class PhoneRates():
    def __init__(self, local_region=None):
        self.local_region = local_region
        self._countries = self.load_countries()

        resource = pkg_resources.resource_filename(__name__, 'resources/codes_and_destinations.csv')
        self._trie = trie.trie()
        self._destinations = dict()
        
        with open(resource, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=";")
            next(reader)
            for row in reader:
                code = str(row[0])
                destination = row[1]
                
                self._destinations[code] = destination
                self._trie[code] = 1

                
        self._groups = self.load_groups()
        self._rates = self.load_rates()
        self._rate_groups = get_groups([val for key, val in self._rates.items()])

    def load_rates(self):
        rates = {}
        resource = pkg_resources.resource_filename(__name__, 'resources/rates.csv')
        with open(resource, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=";")
            next(reader)
            for row in reader:
                destination = row[0]
                group = row[1]
                rate = float(row[2])
                rates[destination, group] = rate

        return rates

    def load_groups(self):
        groups = []
        resource = pkg_resources.resource_filename(__name__, 'resources/groups.csv')
        with open(resource, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=";")
            next(reader)
            for row in reader:
                group = row[0]
                code = str(row[1])
                valid_from = dateutil.parser.parse(row[2])
                groups.append([group, code, valid_from])

        return groups

    def load_countries(self):
        countries = {}
        if len(countries) > 0:
            return countries
        for country in pycountry.countries:
            countries[country.alpha_2] = country.name
        return countries

    def get_group(self, number):
        code = match(number, self._trie)
        if code is None:
            return ""

        groups = [row for row in self._groups if row[1] == code]
        if len(groups) == 0:
            return ""
        
        _groups = sorted(groups, key=itemgetter(2), reverse=True)
        return _groups[0][0]

    def get_rate(self, destination, group):
        rate = self._rates.get((destination, group), 0.0)
        if rate == 0.0:
            return self._rates.get((destination, ''), 0.0)
        else:
            # TODO: Add unit test for non empty group
            return rate
 
    def _parse_number(self, number):
        y = phonenumbers.parse(number, self.local_region)
    
        #if phonenumbers.is_possible_number(y) == False:
        #    print("Isn't possible number:", number)
        #if phonenumbers.is_valid_number(y) == False:
        #    print("Isn't valid number:", number)
    
        number = phonenumbers.format_number(
            y,
            phonenumbers.PhoneNumberFormat.E164,
        )
        region = geocoder.region_codes_for_country_code(y.country_code)[0]
        international = False
        mobile = False
        premium = False
    
        if PhoneNumberType.MOBILE == phonenumbers.number_type(y):
            mobile = True
        if PhoneNumberType.PREMIUM_RATE == phonenumbers.number_type(y):
            premium = True
        if region != self.local_region:
            international = True
    
        return {
            'premium': premium,
            'mobile': mobile,
            'international': international,
            'phonenumber': number,
            'region': region,
            'country': self._countries.get(region, 'INVALID'),
        }

    def parse_number(self, number):
        try:
            return self._parse_number(number)
        except phonenumbers.phonenumberutil.NumberParseException as exn:
            logging.error("invalid number: %s", number)
            return {
                'premium': False,
                'mobile': False,
                'international': False,
                'phonenumber': number,
                'region': 'INVALID',
                'country': 'INVALID'
            }

    def get_stats2(self, calling, called):
        origin = self.parse_number(calling)
        destination = self.parse_number(called)
        stats = self.get_stats(origin['phonenumber'][1:], destination['phonenumber'][1:])
        stats.update({"origin_{}".format(key): val for key, val in origin.items()})
        stats.update({"destination_{}".format(key): val for key, val in destination.items()})
        return stats

    def get_stats(self, calling, called):
        code = match(called, self._trie)
        if code is None:
            return {
                'destination': 'NONE',
                'code': 'NONE',
                'group': 'NONE',
                'rate': 0.0,
                'category': 0,
            }
        else:
            destination = self._destinations[code]

        group = self.get_group(calling)
        rate = self.get_rate(destination, group)
        category = 0
        for j, (_min, _max) in enumerate(self._rate_groups):
            if rate >= _min and rate < _max:
                category = j

        return {
            'destination': destination,
            'code': code,
            'group': group,
            'rate': rate,
            'category': category,
        }


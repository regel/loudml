
import pkg_resources
import csv

# biopython 1.71
from Bio import trie
from Bio.triefind import match

pseudo_prices = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'Z': 10,
}

class PhoneRates():
    def __init__(self):
        resource = pkg_resources.resource_filename(__name__, 'resources/phone_rates.csv')
        self._trie = trie.trie()
        self._dict = dict()
        
        with open(resource, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=";")
            next(reader)
            for row in reader:
                word = row[2]
                fraud_level=row[3]
                pricing = pseudo_prices[fraud_level]
                self._dict[word] = (fraud_level, pricing,)
                self._trie[word] = 1

    def get_fraud_level_and_rate(self, number):
        word = match(number, self._trie)
        if word is None:
            fraud_level = 'Z'
            pricing = 1
        else:
            fraud_level, pricing = self._dict[word]

        return fraud_level, pricing


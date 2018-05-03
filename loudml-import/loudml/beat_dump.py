import pkg_resources

import dateutil.parser
import logging
import json

from rmn_common.data_import import Parser

class BeatParser(Parser):
    def get_template(self, db_name, measurement):
        resource = pkg_resources.resource_filename(__name__, 'resources/beat.template')
        content = open(resource, 'rU').read()
        return content.format(db_name, measurement)

    def decode(self, row):
        try:
            ts = dateutil.parser.parse(row['@timestamp'])
        except (ValueError, TypeError):
            logging.error("got invalid '@timestamp': '%s'", 0)
            print(row)
            raise ValueError

        tag_dict = {
            'Host': row.pop('Host'),
        }

        row.pop('@timestamp')
        data = row
        return int(ts.timestamp()), tag_dict, data

    def read_csv(self, fp, encoding):
        for line in fp: 
            row = json.loads(line.decode('utf-8')) 
            row = row['_source']
            for dt_field in ['timestamp_resp', 'x_timestamp_req', 'x_timestamp_resp']: 
                if dt_field in row: 
                    row[dt_field] = int(1000 * dateutil.parser.parse(row[dt_field]).timestamp())

            row['x_timestamp'] = int(1000 * dateutil.parser.parse(row['@timestamp']).timestamp())
            yield self.decode(row)


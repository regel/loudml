import pkg_resources

import dateutil.parser
import logging
import pandas as pd

from rmn_common.data_import import Parser

class GreenflowParser(Parser):
    def get_template(self, db_name, measurement):
        resource = pkg_resources.resource_filename(__name__, 'resources/greenflow.template')
        content = open(resource, 'rU').read()
        return content.format(db_name, measurement)

    def decode(self, row):
        try:
            ts = dateutil.parser.parse(row['DT'])
        except (ValueError, TypeError):
            logging.error("got invalid 'DT': '%s'", 0)
            print(row)
            raise ValueError


        tag_dict = {}
        data = row.to_dict()
        data.pop('DT')
        return int(ts.timestamp()), tag_dict, data

    def read_csv(self, fp, encoding):
        df = pd.read_csv(
            fp,
            encoding=encoding,
            delimiter=';',
            decimal=",",
            dtype={
                'Lys': float,
                'Noise': float,
                'Relative humidity': float,
                'Temperature': float,
                'Volume': float,
                'NrP': int,
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

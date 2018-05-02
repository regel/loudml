"""
Module for importing data in Datasources
"""

import argparse
import chardet
import datetime
import gzip
import tarfile
import logging
import os
import time
import pkg_resources

from abc import (
    ABCMeta,
    abstractmethod,
)

from loudml.datasource import (
    DataSource,
    load_datasource,
)

from loudml import (
    errors,
)


class Parser(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.current_file_path = None
        self.last_progress = time.time()

    @abstractmethod
    def read_csv(self, fp, encoding):
        """
        Read CSV file
        """

    @abstractmethod
    def get_template(self, db_name, measurement='generic'):
        """
        Get Elasticsearch template file
        """

    def show_progress(self, nb_read, progress):
        now = time.time()

        if now - self.last_progress < 15:
            return

        logging.info("%s: %d read entries, %.2f%%",
                     self.current_file_path, nb_read, 100 * progress)
        self.last_progress = now

    def process_csv_stream(self, fp):
        rawdata = fp.read(1000000)
        encoding = chardet.detect(rawdata)
        fp.seek(0)
        return self.read_csv(fp, encoding['encoding'])

    def process_csv(self, path):
        logging.info("processing CSV file: %s", path)
        self.current_file_path = path
        with open(path, 'rb') as fp:
            for data in self.process_csv_stream(fp):
                yield data

    def process_tgz(self, path):
        logging.info("processing compressed file: %s", path)
        self.current_file_path = path
        tar = tarfile.open(path)
        for member in tar.getmembers():
            if member.isfile():
                with tar.extractfile(member) as fp:
                    for data in self.read_csv(fp, None):
                        yield data
        tar.close()

    def process_gzip(self, path):
        logging.info("processing compressed file: %s", path)
        self.current_file_path = path
        with gzip.open(path, 'rb') as fp:
            for data in self.process_csv_stream(fp):
                yield data

    def process_dir(self, path):
        logging.info("processing directory: %s", path)

        for filename in sorted(os.listdir(path)):
            if filename.endswith('.csv'):
                data = self.process_csv(os.path.join(path, filename))
            elif filename.endswith('.csv.gz'):
                data = self.process_gzip(os.path.join(path, filename))
            else:
                logging.warning("ignoring unknown file: %s",
                                os.path.join(path, filename))
                continue

            for doc in data:
                yield doc

    def run(self, path):
        """
        Run parser on given path
        """
        if path.endswith('.csv'):
            return self.process_csv(path)
        elif path.endswith('.csv.gz'):
            return self.process_gzip(path)
        elif path.endswith('.tgz') or path.endswith('.tar.gz'):
            return self.process_tgz(path)
        elif os.path.isdir(path):
            return self.process_dir(path)
        raise errors.UnsupportedFormat('unknown file type')


def load_parser(format_name):
    """
    Get parser by format name
    """

    for ep in pkg_resources.iter_entry_points('loudml.import_parsers'):
        if ep.name == format_name:
            return ep.load()()

    raise errors.UnsupportedFormat(format_name)


def main():
    """
    Import data into a given Datasource
    """

    parser = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-a', '--addr',
        help="Backend address",
        type=str,
        required=True,
    )
    parser.add_argument(
        '-d', '--database',
        help="Database, or index name",
        type=str,
        required=True,
    )
    parser.add_argument(
        '-f', '--format',
        help="Input format",
        type=str,
        required=True,
    )
    parser.add_argument(
        '-t', '--db_type',
        help="Database type",
        type=str,
        required=True,
    )
    parser.add_argument(
        '-m', '--measurement',
        help="Measurement",
        type=str,
        required=False,
        default='generic',
    )
    parser.add_argument(
        '-F', '--flush',
        help="Flush database",
        action='store_true',
    )
    parser.add_argument(
        'path',
        help="Path to data",
        type=str,
    )
    arg = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    try:
        parser = load_parser(arg.format)
        source = load_datasource({
            'type': arg.db_type,
            'name': 'loudml-import',
            'addr': arg.addr,
            'database': arg.database,
            'index': arg.database,
        })
        if arg.flush:
            delete_db = getattr(source, "delete_db", None)
            if callable(delete_db):
                delete_db()
            delete_index = getattr(source, "delete_index", None)
            if callable(delete_index):
                delete_index()

        create_db = getattr(source, "create_db", None)
        if callable(create_db):
            create_db()

        create_index = getattr(source, "create_index", None)
        if callable(create_index):
            create_index(template_name=arg.database,
                     template=parser.get_template(db_name=arg.database,
                                                  measurement=arg.measurement))

        i = None
        for i, (ts, tag_dict, data) in enumerate(parser.run(arg.path)):
            source.insert_times_data(
                measurement=arg.measurement,
                ts=ts,
                data=data,
                tags=tag_dict,
            )

        if i == None:
            logging.warning("no data read")
        else:
            logging.info("imported %d item(s)", i + 1)

        source.commit()

    except KeyboardInterrupt:
        logging.warning("import cancelled")
        return 1

    return 0

if __name__ == "__main__":
    main()



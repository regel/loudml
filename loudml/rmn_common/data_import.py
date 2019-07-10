"""
Module for loading data from file
"""

import chardet
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
        elif path.endswith(('.tgz', '.tar.gz')):
            return self.process_tgz(path)
        elif os.path.isdir(path):
            return self.process_dir(path)
        raise errors.UnsupportedFormat('unknown file type')


def init_parser(format_name):
    """
    Get parser by format name
    """

    for ep in pkg_resources.iter_entry_points('rmn_import.parsers'):
        if ep.name == format_name:
            return ep.load()()

    raise KeyError("unkown format name `{}`".format(format_name))

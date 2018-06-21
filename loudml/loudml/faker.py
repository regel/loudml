import loudml.vendor

import argparse
import datetime
import logging
import random
import time

from . import (
    errors,
)
from .config import (
    load_config,
)
from .datasource import (
    load_datasource,
)
from .misc import (
    make_datetime,
)
from .randevents import (
    CamelEventGenerator,
    LoudMLEventGenerator,
    FlatEventGenerator,
    SawEventGenerator,
    SinEventGenerator,
)

def generate_data(ts_generator, from_date, to_date):
    for ts in ts_generator.generate_ts(from_date, to_date, step=60):
        yield ts, {
            'foo': random.lognormvariate(10, 1),
        }

def dump_to_json(generator):
    import json

    data = []

    for ts, entry in generator:
        entry['timestamp'] = ts
        data.append(entry)

    print(json.dumps(data,indent=4))

def build_tag_dict(tags=None):
    tag_dict = {}
    if tags:
        for tag in tags.split(','):
            k, v = tag.split(':')
            tag_dict[k] = v
    return tag_dict

def init_datasource(arg, tags=None):
    config = load_config(arg.config)
    src_settings = config.get_datasource(arg.output)
    datasource = load_datasource(src_settings)

    if arg.clear:
        datasource.drop()

    kwargs = {}

    if src_settings['type'] == 'elasticsearch':
        properties = {
            "timestamp": {
                "type": "date"
            },
            "foo": {
                "type": "float",
            },
        }

        if tags:
            for k in tags.keys():
                properties[k] = {
                    "type": "keyword",
                }

        kwargs['template_name'] = "faker"
        kwargs['template'] = {
            'template': "faker",
            'mappings': {
                arg.doc_type: {
                    'include_in_all': True,
                    'properties': properties,
                },
            }
        }

    datasource.init(**kwargs)
    return datasource

def dump_to_datasource(generator, datasource, tags=None, **kwargs):
    for ts, data in generator:
        now = time.time()
        if ts > now:
            time.sleep(ts - now)

        datasource.insert_times_data(
            ts=ts,
            data=data,
            tags=tags,
            **kwargs
        )

def main():
    """
    Generate dummy data
    """

    parser = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        default="/etc/loudml/config.yml",
        help="Path to LoudML configuration",
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help="Name of destination datasource",
    )
    parser.add_argument(
        '-m', '--measurement',
        help="Measurement",
        type=str,
        default='dummy_data',
    )
    parser.add_argument(
        '--doc-type',
        help="Document type",
        type=str,
        default='generic',
    )
    parser.add_argument(
        '--from',
        help="From date",
        type=str,
        default="now-7d",
        dest='from_date',
    )
    parser.add_argument(
        '--to',
        help="To date",
        type=str,
        default="now",
        dest='to_date',
    )
    parser.add_argument(
        '--shape',
        help="Data shape",
        choices=['flat', 'saw', 'sin', 'camel', 'loudml'],
        default='sin',
    )
    parser.add_argument(
        '--amplitude',
        help="Peak amplitude for periodic shapes",
        type=float,
        default=1,
    )
    parser.add_argument(
        '--base',
        help="Base value for number of events",
        type=float,
        default=1,
    )
    parser.add_argument(
        '--trend',
        help="Trend (event increase per hour)",
        type=float,
        default=0,
    )
    parser.add_argument(
        '--clear',
        help="Clear database or index before insertion "\
             "(risk of data loss! Use with caution!)",
        action='store_true',
    )
    parser.add_argument(
        '--tags',
        help="Tags",
        type=str,
    )

    arg = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    tags = build_tag_dict(arg.tags)

    if arg.output:
        try:
            datasource = init_datasource(arg, tags=tags)
        except errors.LoudMLException as exn:
            logging.error(exn)
            return 1

    if arg.shape == 'flat':
        ts_generator = FlatEventGenerator(base=arg.base, trend=arg.trend)
    elif arg.shape == 'loudml':
        ts_generator = LoudMLEventGenerator(base=arg.base, trend=arg.trend)
    elif arg.shape == 'camel':
        ts_generator = CamelEventGenerator(base=arg.base, trend=arg.trend)
    elif arg.shape == 'saw':
        ts_generator = SawEventGenerator(
            base=arg.base,
            amplitude=arg.amplitude,
            trend=arg.trend,
        )
    else:
        ts_generator = SinEventGenerator(
            base=arg.base,
            amplitude=arg.amplitude,
            trend=arg.trend,
            sigma=2,
        )

    from_date = make_datetime(arg.from_date)
    to_date = make_datetime(arg.to_date)

    logging.info("generating data from %s to %s", from_date, to_date)

    generator = generate_data(ts_generator, from_date.timestamp(), to_date.timestamp())

    if arg.output is None:
        dump_to_json(generator)
    else:
        kwargs = {}

        if arg.measurement:
            kwargs['measurement'] = arg.measurement
        if arg.doc_type:
            kwargs['doc_type'] = arg.doc_type

        try:
            dump_to_datasource(generator, datasource, tags=tags, **kwargs)
        except errors.LoudMLException as exn:
            logging.error(exn)

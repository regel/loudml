import loudml.vendor

import argparse
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
    TriangleEventGenerator,
)

def generate_data(
    ts_generator,
    from_date,
    to_date,
    step_ms,
    errors,
    burst_ms,
    field,
):
    ano = False
    previous_ts = None
    for ts in ts_generator.generate_ts(
        from_date,
        to_date,
        step_ms=step_ms,
    ):
        if ano == False and errors > 0:
            val = random.random()
            if val < errors:
                ano = True
                total_burst_ms = 0
                previous_ts = ts

        if ano == True and total_burst_ms < burst_ms:
            total_burst_ms += (ts - previous_ts) * 1000.0
            previous_ts = ts
        else:
            ano = False
            yield ts, {
                field: random.lognormvariate(10, 1),
            }

def dump_to_json(generator):
    import json

    data = []

    for ts, entry in generator:
        entry['timestamp'] = ts
        data.append(entry)

    print(json.dumps(data, indent=4))


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
            "value": {
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
        help="Path to Loud ML configuration",
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
        '--field',
        help="Field",
        type=str,
        default="value",
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
        choices=[
            'flat',
            'saw',
            'sin',
            'triangle',
            'camel',
            'loudml',
        ],
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
        '--period',
        help="Period in seconds",
        type=float,
        default=24 * 3600,
    )
    parser.add_argument(
        '--sigma',
        help="Sigma",
        type=float,
        default=2,
    )
    parser.add_argument(
        '--step-ms',
        help="Milliseconds elapsed in each step fo generating samples",
        type=int,
        default=60000,
    )
    parser.add_argument(
        '-e', '--errors',
        help="Output anomalies with the given error rate",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        '-b', '--burst-ms',
        help="Burst duration, for anomalies",
        type=int,
        default=0,
    )
    parser.add_argument(
        '--clear',
        help="Clear database or index before insertion "
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
        ts_generator = CamelEventGenerator(
            base=arg.base,
            amplitude=arg.amplitude,
            period=arg.period,
            trend=arg.trend,
            sigma=arg.sigma,
        )
    elif arg.shape == 'saw':
        ts_generator = SawEventGenerator(
            base=arg.base,
            amplitude=arg.amplitude,
            period=arg.period,
            trend=arg.trend,
            sigma=arg.sigma,
        )
    elif arg.shape == 'triangle':
        ts_generator = TriangleEventGenerator(
            base=arg.base,
            amplitude=arg.amplitude,
            period=arg.period,
            trend=arg.trend,
            sigma=arg.sigma,
        )
    else:
        ts_generator = SinEventGenerator(
            base=arg.base,
            amplitude=arg.amplitude,
            trend=arg.trend,
            period=arg.period,
            sigma=arg.sigma,
    )

    from_date = make_datetime(arg.from_date)
    to_date = make_datetime(arg.to_date)

    logging.info("generating data from %s to %s", from_date, to_date)

    generator = generate_data(
        ts_generator,
        from_date.timestamp(),
        to_date.timestamp(),
        arg.step_ms,
        arg.errors,
        arg.burst_ms,
        arg.field,
    )

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

import argparse
import logging

from rmn_common.data_import import (
    init_parser,
)

from loudml.datasource import (
    load_datasource,
)

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
        '--doc-type',
        help="Document type",
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
        parser = init_parser(arg.format)
        source = load_datasource({
            'type': arg.db_type,
            'name': 'loudml-import',
            'addr': arg.addr,
            'database': arg.database,
            'index': arg.database,
        })

        if arg.flush:
            source.drop()

        kwargs = {}

        template = parser.get_template(arg.database, arg.measurement)
        if template:
            kwargs['template_name'] = arg.database
            kwargs['template'] = template

        source.init(**kwargs)

        kwargs = {}

        if arg.measurement:
            kwargs['measurement'] = arg.measurement
        if arg.doc_type:
            kwargs['doc_type'] = arg.doc_type

        i = None
        for i, (ts, tag_dict, data) in enumerate(parser.run(arg.path)):
            source.insert_times_data(
                ts=ts,
                data=data,
                tags=tag_dict,
                **kwargs,
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

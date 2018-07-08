#!/usr/bin/env python3

"""
Setup import module for Loud ML
"""

import os
from setuptools import setup

setup(
    name='loudml-import',

    version=os.getenv('LOUDML_VERSION', '1.4'),

    description="Import module for Loud ML",

    py_modules=[
    ],

    namespace_packages=['loudml'],

    packages=[
        'loudml',
    ],

    setup_requires=[
        'jinja2',
    ],

    tests_require=['nose'],
    test_suite='nose.collector',

    package_data = { '': [
        'codes_and_destinations.csv',
        'groups.csv',
        'rates.csv',
        'phonedb.template',
        'greenflow.template',
        'beat.template',
    ]},

    install_requires=[
        'loudml',
        'pandas',
        'phonenumbers',
        'pycountry',
        'biopython',
    ],
    entry_points={
        'console_scripts': [
            'loudml-import=loudml.import_tool:main',
        ],
        'rmn_import.parsers': [
            'cirpack=loudml.cirpack:CdrParser',
            'paritel=loudml.paritel:CdrParser',
            'greenflow=loudml.greenflow:GreenflowParser',
            'beat_dump=loudml.beat_dump:BeatParser',
        ],

    },
    include_package_data=True,
    zip_safe=False,

)

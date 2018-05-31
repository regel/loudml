#!/usr/bin/env python3

"""
Setup InfluxDB module for LoudML
"""

import os
from setuptools import setup

setup(
    name='loudml-influx',

    version=os.getenv('LOUDML_VERSION', '1.3'),

    description="InfluxDB module for LoudML",

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

    install_requires=[
        'loudml',
        'influxdb>=5.0.0',
    ],

    include_package_data=True,
    zip_safe=False,

    entry_points={
        'loudml.datasources': [
            'influxdb=loudml.influx:InfluxDataSource',
        ],
    },
)

#!/usr/bin/env python3

"""
Setup InfluxDB module for LoudML
"""

from setuptools import setup

setup(
    name='loudml-influx',

    version='0.1.0',

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
        'influxdb',
    ],

    include_package_data=True,
    zip_safe=False,

    entry_points={
        'loudml.datasources': [
            'influxdb=loudml.influx:InfluxDataSource',
        ],
    },
)

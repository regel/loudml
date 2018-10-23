#!/usr/bin/env python3

"""
Setup Elasticsearch module for Loud ML
"""

import os
from setuptools import setup

setup(
    name='loudml-elastic',

    version=os.getenv('LOUDML_VERSION', '1.4'),

    description="Elasticsearch module for Loud ML",

    py_modules=[
    ],

    namespace_packages=['loudml'],

    packages=[
        'loudml',
    ],

    setup_requires=[
    ],

    tests_require=['nose'],
    test_suite='nose.collector',

    install_requires=[
        'loudml',
        'elasticsearch',
    ],

    include_package_data=True,
    zip_safe=False,

    entry_points={
        'loudml.datasources': [
            'elasticsearch=loudml.elastic:ElasticsearchDataSource',
        ],
    },
)

#!/usr/bin/env python3

"""
Setup Elasticsearch AWS module for Loud ML
"""

import os
from setuptools import setup

setup(
    name='loudml-elastic-aws',

    version=os.getenv('LOUDML_VERSION', '1.3'),

    description="Elasticsearch AWS module for Loud ML",

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
        'loudml-elastic',
        'boto3',
        'requests-aws4auth',
    ],

    include_package_data=True,
    zip_safe=False,

    entry_points={
        'loudml.datasources': [
            'elasticsearch_aws=loudml.elastic_aws:ElasticsearchAWSDataSource',
        ],
    },
)

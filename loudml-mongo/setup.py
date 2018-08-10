#!/usr/bin/env python3

"""
Setup MongoDB module for LoudML
"""

import os
from setuptools import setup

setup(
    name='loudml-mongo',

    version=os.getenv('LOUDML_VERSION', '1.3'),

    description="MongoDB module for LoudML",

    py_modules=[
    ],

    namespace_packages=['loudml'],

    packages=[
        'loudml',
    ],

    setup_requires=[
        'pymongo',
    ],

    tests_require=['nose'],
    test_suite='nose.collector',

    install_requires=[
        'loudml',
        'pymongo',
    ],

    include_package_data=True,
    zip_safe=False,

    entry_points={
        'loudml.datasources': [
            'mongodb=loudml.mongo:MongoDataSource',
        ],
    },
)

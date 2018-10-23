#!/usr/bin/env python3

"""
Setup Warp10 module for Loud ML
"""

import os
from setuptools import setup

setup(
    name='loudml-warp10',

    version=os.getenv('LOUDML_VERSION', '1.3'),

    description="Warp10 module for Loud ML",

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
        'warp10client',
    ],

    include_package_data=True,
    zip_safe=False,

    entry_points={
        'loudml.datasources': [
            'warp10=loudml.warp10:Warp10DataSource',
        ],
    },
)

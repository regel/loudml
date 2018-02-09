#!/usr/bin/env python3

"""
Setup iVOIP module for LoudML
"""

from setuptools import setup

setup(
    name='loudml-ivoip',

    version='0.1.0',

    description="iVOIP module for LoudML",

    py_modules=[
    ],

    namespace_packages=['loudml'],

    packages=[
        'loudml',
    ],

    tests_require=['nose'],
    test_suite='nose.collector',

    install_requires=[
        'loudml',
        'loudml-elastic',
    ],

    include_package_data=True,
    zip_safe=False,

    entry_points={
        'loudml.datasources': [
            'ivoip=loudml.ivoip:IVoipDataSource',
        ],
        'loudml.models': [
            'ivoip_fingerprints=loudml.ivoip:IVoipFingerprintsModel',
        ]
    },
)

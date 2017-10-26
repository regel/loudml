#!/usr/bin/env python3

"""
Setup BONSAI ML python package
"""

from setuptools import setup
from os import path

setup(
    name='loudml',

    version='0.1.0',

    description='BONSAI ML python package',

    py_modules=[
    ],

    packages=[
        'loudml',
    ],

    setup_requires=[
        'jinja2',
    ],

    tests_require=['nose'],
    test_suite='nose.collector',

    install_requires=[
        'flask',
        'elasticsearch',
        'tensorflow',
        'h5py',
        'hyperopt',
        'Pillow',
    ],

    extras_require={
    },

    package_data={
    },

    data_files=[
    ],

    include_package_data=True,
    zip_safe=False,

    entry_points={
        'console_scripts': [
            'loudmld=loudml.server:main',
            'loudml_times=loudml.times:main',
            'loudml_ivoip=loudml.ivoip:main',
        ],
    },
)

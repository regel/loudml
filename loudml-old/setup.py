#!/usr/bin/env python3

"""
Setup old loudml package
"""

from setuptools import setup
from os import path

setup(
    name='loudml-old',

    version='0.1.0',

    description='BONSAI ML python package',

    py_modules=[
    ],

    packages=[
        'loudml_old',
    ],

    setup_requires=[
        'jinja2',
    ],

    tests_require=['nose'],
    test_suite='nose.collector',

    install_requires=[
        'dateutils',
        'flask',
        'elasticsearch',
        'tensorflow',
        'h5py',
        'hyperopt',
        'Pillow',
        'PyYAML',
        'requests>=2.17.0',
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
            'loudmld_old=loudml_old.server:main',
            'loudml_times=loudml_old.times:main',
            'loudml_ivoip=loudml_old.ivoip:main',
        ],
    },
)

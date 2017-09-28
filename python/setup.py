#!/usr/bin/env python3

"""
Setup BONSAI ML python package
"""

from setuptools import setup
from os import path

setup(
    name='bonsai',

    version='0.1.0',

    description='BONSAI ML python package',

    py_modules=[
    ],

    packages=[
        'bonsai',
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
            'bonsaid=bonsai.server:main',
            'bonsai_series=bonsai.compute:main',
            'bonsai_segmap=bonsai.nnsom:main',
        ],
    },
)

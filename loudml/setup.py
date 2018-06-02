#!/usr/bin/env python3

"""
Setup Loud ML python package
"""

import os
from setuptools import setup

setup(
    name='loudml',

    version=os.getenv('LOUDML_VERSION', '1.4'),

    description="Machine Learning application",

    py_modules=[
    ],

    packages=[
        'loudml',
        'rmn_common',
    ],

    setup_requires=[
    ],

    tests_require=['nose'],
    test_suite='nose.collector',

    install_requires=[
        "dictdiffer>=0.7.1",
        'uuid',
        "dateutils>=0.6.6",
        "Flask>=0.12.1",
        "Flask-restful>=0.3.5",
        "gevent>=1.3.1",
        "loudml-api>=1.3.0",
        "networkx==1.11",
        "tensorflow==1.3.0",
        "h5py>=2.7.0",
        "hyperopt==0.1",
        "numpy>=1.10.0",
        "Pebble>=4.3.8",
        "psutil>=2.2.1",
        "pycrypto>=2.6.1",
        "PyYAML>=3.11",
        "voluptuous>=0.10.5",
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
            'loudml-faker=loudml.faker:main',
            'loudml=loudml.cli:main',
            'loudml-lic=loudml.license_mgmnt:main',
        ],
        'loudml.commands': [
            'create-model=loudml.cli:CreateModelCommand',
            'delete-model=loudml.cli:DeleteModelCommand',
            'list-models=loudml.cli:ListModelsCommand',
            'show-model=loudml.cli:ShowModelCommand',
            'train=loudml.cli:TrainCommand',
            'predict=loudml.cli:PredictCommand',
            'forecast=loudml.cli:ForecastCommand',
            'run=loudml.cli:RunCommand',
        ],
        'loudml.models': [
            'fingerprints=loudml.fingerprints:FingerprintsModel',
            'timeseries=loudml.timeseries:TimeSeriesModel',
        ],
        'loudml.hooks': [
            'annotations=loudml.annotations:AnnotationHook',
        ],
    },
)

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
        'python-crontab==2.3.5',
        "dateutils>=0.6.6",
        "Flask>=0.12.1",
        "Flask-restful>=0.3.5",
        "gevent>=1.3.1",
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
        "scikit-learn>=0.19.1",
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
            'list-checkpoints=loudml.cli:ListCheckpointsCommand',
            'save-checkpoint=loudml.cli:SaveCheckpointCommand',
            'load-checkpoint=loudml.cli:LoadCheckpointCommand',
            'create-model=loudml.cli:CreateModelCommand',
            'delete-model=loudml.cli:DeleteModelCommand',
            'list-models=loudml.cli:ListModelsCommand',
            'show-model=loudml.cli:ShowModelCommand',
            'train=loudml.cli:TrainCommand',
            'predict=loudml.cli:PredictCommand',
            'forecast=loudml.cli:ForecastCommand',
            'plot=loudml.cli:PlotCommand',
        ],
        'loudml.models': [
            'donut=loudml.donut:DonutModel',
        ],
        'loudml.hooks': [
            'annotations=loudml.annotations:AnnotationHook',
        ],
        'loudml.datasources': [
            'influxdb=loudml.influx:InfluxDataSource',
            'elasticsearch=loudml.elastic:ElasticsearchDataSource',
            'elasticsearch_aws=loudml.elastic_aws:ElasticsearchAWSDataSource',
            'warp10=loudml.warp10:Warp10DataSource',
            'mongodb=loudml.mongo:MongoDataSource',
        ],
    },
)

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
# DO NOT ADD REQUIRES HERE
# See base/vendor/requirements.txt.in
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
        ],
        'loudml.commands': [
            'list-checkpoints=loudml.cli:ListCheckpointsCommand',
            'save-checkpoint=loudml.cli:SaveCheckpointCommand',
            'load-checkpoint=loudml.cli:LoadCheckpointCommand',
            'create-model=loudml.cli:CreateModelCommand',
            'delete-model=loudml.cli:DeleteModelCommand',
            'list-models=loudml.cli:ListModelsCommand',
            'list-templates=loudml.cli:ListTemplatesCommand',
            'show-model=loudml.cli:ShowModelCommand',
            'load-data=loudml.cli:LoadDataCommand',
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

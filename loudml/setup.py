#!/usr/bin/env python3

"""
Setup LoudML python package
"""

from setuptools import setup

setup(
    name='loudml',

    version='0.1.0',

    description="Machine Learning application",

    py_modules=[
    ],

    packages=[
        'loudml',
    ],

    setup_requires=[
    ],

    tests_require=['nose'],
    test_suite='nose.collector',

    install_requires=[
        "dateutils>=0.6.6",
        "Flask>=0.12.2",
        "Flask-restful>=0.3.6",
        "networkx==1.11",
        "tensorflow >=1.3.0, <=1.3.1",
        "h5py==2.7.1",
        "hyperopt==0.1",
        "psutil>=2.2.1",
        "PyYAML>=3.11",
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
            'create-model=loudml.cli:CreateModelCommand',
            'delete-model=loudml.cli:DeleteModelCommand',
            'list-models=loudml.cli:ListModelsCommand',
            'show-model=loudml.cli:ShowModelCommand',
            'train=loudml.cli:TrainCommand',
            'predict=loudml.cli:PredictCommand',
        ],
        'loudml.models': [
            'timeseries=loudml.timeseries:TimeSeriesModel',
        ],
    },
)

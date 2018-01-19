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
        'dateutils',
        'flask',
        'flask_restful',
        'tensorflow',
        'h5py',
        'hyperopt',
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
            'loudmld=loudml.server:main',
            'loudml-faker=loudml.faker:main',
            'loudml=loudml.cli:main',
        ],
        'loudml.commands': [
            'create-model=loudml.cli:CreateModelCommand',
            'delete-model=loudml.cli:DeleteModelCommand',
            'list-models=loudml.cli:ListModelsCommand',
            'train=loudml.cli:TrainCommand',
            'predict=loudml.cli:PredictCommand',
        ],
        'loudml.models': [
            'timeseries=loudml.times:TimesModel',
        ],
    },
)

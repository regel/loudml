#!/usr/bin/env python3

"""
Setup Loud ML python package
"""

import os
from io import open as io_open
from setuptools import setup


def find_version():
    """Get version from loudml/_version.py"""
    _locals = locals()
    src_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(src_dir, 'loudml', '_version.py')
    with io_open(version_file, mode='r') as fd:
        exec(fd.read())  # __version__ is set in the exec call.
        return _locals['__version__']


setup(
    name='loudml',
    version=find_version(),

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
        ],
        'loudml.models': [
            'donut=loudml.donut:DonutModel',
        ],
        'loudml.hooks': [
            'annotations=loudml.annotations:AnnotationHook',
        ],
        'loudml.buckets': [
            'influxdb=loudml.influx:InfluxBucket',
            'elasticsearch=loudml.elastic:ElasticsearchBucket',
            'elasticsearch_aws=loudml.elastic_aws:ElasticsearchAWSBucket',
            'warp10=loudml.warp10:Warp10Bucket',
            'mongodb=loudml.mongo:MongoBucket',
        ],
    },
)

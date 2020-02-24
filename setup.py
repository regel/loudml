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
        'Jinja2>=2.9',
        'Flask-restful>=0.3.6',
        'Flask>=0.12.2',
        'pebble>=4.3.8',
        'gevent>=1.3.1',
        'PyYAML>=3.11',
        'schedule>=0.6.0',
        'requests>=2.14.0',
        'pytz>=2019.2',
        'dateutils>=0.6.6',
        'h5py==2.9.0',
        'hyperopt>=0.1',
        'networkx==2.2',
        'numpy==1.16.4',
        'pycrypto>=2.6.1',
        'voluptuous==0.10.5',
        'dictdiffer>=0.7.1',
        'elasticsearch==6.3.1',
        'boto3>=1.7.58',
        'requests-aws4auth>=0.9',
        'influxdb>=5.0.0',
        'pymongo',
        'warp10client @ git+git://github.com/regel/python-warp10client.git',
        'daiquiri>=1.5.0',
        'loudml-python>=1.6.0,<2.0.0',
    ],

    extras_require={
        'interactive': ['matplotlib==3.0.3'],
        'none': [],
        'cpu': ['tensorflow==1.13.2'],
        'gpu': ['tensorflow-gpu==1.13.2'],
        'dev': [
            'autopep8>=1.5',
            'flake8>=3.7.9',
            'nose>=1.3.7',
            'coverage>=5.0.3',
        ],
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
    },
)

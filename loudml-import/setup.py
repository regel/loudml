#!/usr/bin/env python3

"""
Setup import module for LoudML
"""

from setuptools import setup

setup(
    name='loudml-import',

    version='1.2',

    description="Import module for LoudML",

    py_modules=[
    ],

    namespace_packages=['loudml'],

    packages=[
        'loudml',
    ],

    setup_requires=[
        'jinja2',
    ],

    tests_require=['nose'],
    test_suite='nose.collector',

    package_data = { '': [
        'paritel.template',
        'cirpack.template',
        'greenflow.template',
        'beat.template',
    ]},

    install_requires=[
        'loudml',
        'pandas',
        'phonenumbers',
    ],
    entry_points={
        'console_scripts': [
            'loudml-import=loudml.import_tool:main',
        ],
        'rmn_import.parsers': [
            'cirpack=loudml.cirpack:CdrParser',
            'paritel=loudml.paritel:CdrParser',
            'greenflow=loudml.greenflow:GreenflowParser',
            'beat_dump=loudml.beat_dump:BeatParser',
        ],

    },
    include_package_data=True,
    zip_safe=False,

)

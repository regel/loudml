#!/usr/bin/env python3

# This file is part of LoudML Plug-In API. LoudML Plug-In API is free software:
# you can redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Red Mint Network

"""
Setup LoudML plug-in API
"""

from setuptools import setup

setup(
    name='loudml-api',

    version='1.4.2',

    description="Package providing the Python LoudML plug-in API",

    author="Red Mint Network",
    author_email="contact@loudml.com",
    url="http://loudml.com",
    license="GNU GPL v2",

    namespace_packages=['loudml'],

    packages=[
        'loudml',
    ],

    install_requires=[
    ],
)

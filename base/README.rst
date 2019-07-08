This directory contains tools to package Loud ML dependencies.

Requirements
------------

Building the packages requires a recent version of Python PIP. Because the
version packaged in EPEL is not recent enough, the program has to be
installed by hand in an exact location.

.. code-block::

    # virtualenv -p python3 /opt/loudml
    # /opt/loudml/bin/pip install -U pip
    # /opt/loudml/bin/pip --version

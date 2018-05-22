This directory contains tools to package LoudML dependencies.

Requirements
------------

Building the packages requires a recent version of Python PIP. Because the
version packaged in EPEL is not recent enough, the program has to be
installed by hand in an exact location.

.. code-block::

    # virtualenv -p python3 /opt/redmint
    # /opt/redmint/bin/pip install -U pip
    # /opt/redmint/bin/pip --version
    pip 10.0.1 from /opt/redmint/lib/python3.4/site-packages/pip (python 3.4)

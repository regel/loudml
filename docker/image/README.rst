Dockerfile to build a Loud ML container.

Build Docker container
======================

There are several options as to the source for the packages. They are listed
below.

Buildbot
--------

Use the artifacts generated during the last build in the CI.

Requirements:
- `aws` CLI tool installed and configured
- proper credentials to access the S3 bucket

.. console:

   $ make image repo_src=buildbot


Local packages
--------------

Use the Debian packages at the top of the source directory

Requirements:
- Debian packages must already exist (run ``make deb`` in the top
  directory)

.. console:

   $ make image repo_src=local


Staging AWS S3 bucket
---------------------

Use the packages in the staging bucket in AWS S3

.. console:

   $ make image repo_src=staging


Official packages
-----------------

Use the official packages in the Loud ML AWS S3 bucket.

.. console:

   $ make image repo_src=release


Run Docker container
====================

Use a command such as the one below, where the Docker image is changed to
suit your needs.

.. console:

   $ docker run --rm -ti -p 8077:8077 -v $VOLUME:/var/lib/loudml/models:rw \
                loudml/loudml

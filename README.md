![x](https://raw.githubusercontent.com/regel/loudml/master/donut.png)

# Loud ML - Reveal the hidden

[![CircleCI][1]](https://circleci.com/gh/regel/loudml) [![Docker pulls][2]](https://hub.docker.com/r/loudml/community) [![Coverage][3]](https://sonarcloud.io/dashboard?id=regel_loudml) [![Netlify Status][4]](https://app.netlify.com/sites/flamboyant-cori-981ea4/deploys)

Loud ML is an open source inference engine for metrics and events, and the fastest way to embed machine learning in your time series application. This includes APIs for storing and querying data, processing it in the background for ML or detecting outliers for alerting purposes, and more. 

## Help make this document better

This page, as well as the rest of our docs, are open-source and available on GitHub. We welcome your contributions.

* To report a problem in the documentation, or to submit feedback and comments, please open an issue on GitHub.

## An Open-Source AI Library for Time Series Data

Loud ML is an open source **time series inference engine** built on top of TensorFlow. It's useful to forecast data, detect outliers, and automate your process using future knowledge.

## Features

* Built-in HTTP API that facilitates the integration in other applications
* Data agnostic. The ML engine consumes data from different buckets to achieve seamless data experience. Supported data buckets include:
  - [ElasticSearch](https://github.com/elastic/elasticsearch)
  - [InfluxDB](https://github.com/influxdata/influxdb)
  - [MongoDB](https://github.com/mongodb/mongo)
  - [OpenTSDB](https://github.com/OpenTSDB/opentsdb). Contributed by Volodymyr Sergeyev
* JSON configuration
* Simple to install and manage
* Donut unsupervised learning model [arXiv 1802.03903](https://arxiv.org/abs/1802.03903)
* Data processing in near real-time: data buckets are queried
  at regular intervals and feed the inference engine to return results

## Installation

### Local install

loudmld can be installed using `pip` similar to other Python packages. Do not use `sudo` with `pip`. It is usually good to work in a [virtualenv](https://virtualenv.pypa.io/en/latest/) or [venv](https://docs.python.org/3/library/venv.html) to avoid conflicts with other package managers and Python projects. For a quick introduction see [Python Virtual Environments in Five Minutes](https://bit.ly/py-env>)

Run inside a virtualenv:

```bash
make install
```

## Getting Started

## Running loudmld

You can start the Loud ML model server using:

* `systemctl start loudmld` if you have installed Loud ML using an official Debian or RPM package, and are running a distro with `systemd`.
* `loudmld` if you have built Loud ML from source.

```bash
loudmld -c <path/to/config.yml file>
```

## Running loudml command-line interface

One extra package is needed to run the command line interface.

If you've installed `loudml-python` locally, the `loudml` command should be available via the command line. Executing loudml will start the CLI and automatically connect to the local Loud ML model server instance (assuming you have already started the server with `systemctl start loudmld` or by running loudmld directly).
 
```bash
pip install loudml-python
```

The Python client library is [open source](https://github.com/loudml/loudml-python)

Contributors wanted! Official client libraries for Javascript, Java, Ruby, Go can be found at: https://github.com/loudml

## Running unit tests

```bash
make test
```

## Building Packages

```bash
make clean && make rpm
```

```bash
make clean && make repo
```

## Documentation

* Read more about the [design goals and motivations of the project](http://get.influxdata.com/rs/972-GDU-533/images/CustomerCaseStudy_LoudML.pdf).
* Follow the [getting started guide](https://loudml.io/guide/en/loudml/reference/current/getting-started.html) to learn the basics in just a few minutes.
* Learn more about [Loud ML's key concepts](https://loudml.io/guide/en/loudml/reference/current/glossary.html).

## Contributing

If you're feeling adventurous and want to contribute to Loud ML, see our [contributing doc](https://github.com/regel/loudml/blob/master/CONTRIBUTING.md) for info on how to make feature requests, build from source, and run tests.

## Licensing

See [LICENSE](./LICENSE)

## Looking for Support?

Contact [contact@loudml.io](mailto:contact@loudml.io) to learn how we can best help you succeed.

[1]: https://circleci.com/gh/regel/loudml.svg?style=svg

[2]: https://img.shields.io/docker/pulls/loudml/community.svg

[3]: https://sonarcloud.io/api/project_badges/measure?project=regel_loudml&metric=coverage

[4]: https://api.netlify.com/api/v1/badges/aee0d77f-54ac-413e-bb7f-34a154f47765/deploy-status

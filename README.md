![x](https://raw.githubusercontent.com/regel/loudml/master/donut.png)

# Loud ML - Reveal the hidden

[![CircleCI](https://circleci.com/gh/regel/loudml.svg?style=svg)](https://circleci.com/gh/regel/loudml) [![Docker pulls](https://img.shields.io/docker/pulls/loudml/community.svg)](https://hub.docker.com/r/loudml/community) [![Coverage](https://sonarcloud.io/api/project_badges/measure?project=regel_loudml&metric=coverage)](https://sonarcloud.io/dashboard?id=regel_loudml) [![Netlify Status](https://api.netlify.com/api/v1/badges/aee0d77f-54ac-413e-bb7f-34a154f47765/deploy-status)](https://app.netlify.com/sites/flamboyant-cori-981ea4/deploys)

Loud ML is an open source inference engine for metrics and events, and the fastest way to embed machine learning in your time series application. This includes APIs for storing and querying data, processing it in the background for ML or detecting outliers for alerting purposes, and more. 

## Help make this document better

This page, as well as the rest of our docs, are open-source and available on GitHub. We welcome your contributions.

* To report a problem in the documentation, or to submit feedback and comments, please open an issue on GitHub.

## An Open-Source AI Library for Time Series Data

Loud ML is an open source **time series inference engine** built on top of TensorFlow. It's useful to forecast data, detect outliers, and automate your process using future knowledge.

## Features

* Built-in HTTP API that facilitates the integration in other applications.
* Data agnostic. The ML engine sits on top of all your data stores to provide instant results.
* JSON like model feature specification.
* Simple to install and manage, and fast to get data in and out.
* Donut unsupervised learning model [arXiv 1802.03903](https://arxiv.org/abs/1802.03903)
* It aims to process data in near real-time. That means data is queried
  at regular intervals and feed to the inference engine to return results.

## Installation

We recommend installing Loud ML using one of the [pre-built packages](https://loudml.io/guide/en/loudml/reference/current/install-loudml.html). Then start Loud ML using:

* `systemctl start loudmld` if you have installed Loud ML using an official Debian or RPM package, and are running a distro with `systemd`.
* `loudmld` and `loudml` if you have built Loud ML from source.

### Local install

Inside a virtualenv:

```bash
make install
```

System-wide installation:

```bash
sudo make install
```

## Getting Started

## Running loudml command-line interface

```bash
loudml -c <path/to/configuration> <command>
```

See help for further information about commands

```bash
loudml -h
```

## Running loudmld

```bash
loudmld -c <path/to/configuration>
```

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


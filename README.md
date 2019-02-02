# Loud ML - Reveal the hidden

Loud ML is an AI bot that will enhance the management and operations of your most valuable assets through automation and prediction.

# Getting started for new users

http://loudml.io/guide/[Documentation]

# Getting started for developers

## Installation for developpers

Inside a virtualenv:

```bash
make install
```

System-wide installation:

```bash
sudo make install
```

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
loudml -c <path/to/configuration>
```

## Running unit tests

```bash
make test
```

If your Elasticsearch and InfluxDB servers do not run on locahost, set their
addresses using environment variables:

```bash
ELASTICSEARCH_ADDR="<host>:<port>" INFLUXDB_ADDR="<host>:<port>" make test
```

## Building RPMs

```bash
make clean && make rpm
```

Building RPM repository:

```bash
make clean && make repo
```

## Generate data for testing purposes

The `loudml-core` package includes a `loudml-faker` tool for generating random
data for testing purposes.

For help, see:

```bash
loudml-faker -h
```

### Generating data into InfluxDB

```bash
loudml-faker -o influx -a localhost -b mydb -m mydata --from now-3w --to now
```

### Generating data into Elasticsearch

```bash
loudml-faker -o elastic -a localhost:9200 -i myindex --from now-3w --to now
```


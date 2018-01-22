# LoudML core

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

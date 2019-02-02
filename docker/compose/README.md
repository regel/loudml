# Dockerfile for Loud ML

This directory contains a `Dockerfile` that can be used with Docker Compose
to spawn a TICK-L stack, i.e. a stack of containers with:
- Telegraf
- InfluxDB
- Chronograf for Loud ML
- Kapacitor
- Loud ML Community Edition

The following ports are exposed on the host:
- 8077: Loud ML
- 8086: InfluxDB
- 8888: Chronograf

The configuration files for Loud ML and Telegraf (`./etc/loudml.yml` and
`./etc/telegraf.conf`) are required.

## Instructions

### Configuration

The default configuration creates several subdirectories in the current
directory:
- `./var/chronograf/`: Chronograf data
- `./var/influxdb/`: InfluxDB database
- `./var/kapacitor/`: Kapacitor data
- `./var/loudml/`: Loud ML models

If you want to change those directories, you can edit the file
`docker-compose.yml` to change the lines below:

```yaml
services:
  loudml:
    volumes:
      - ./var/loudml:/var/lib/loudml
    [...]

  influxdb:
    volumes:
      - ./var/influxdb:/var/lib/influxdb
    [...]

  kapacitor:
    volumes:
      - ./var/kapacitor:/var/lib/kapacitor
    [...]
```

### Basic usage

To start all of the containers:

```
$ docker-compose up
```

You can open Chronograf on the host port 8888. This is the same Chronograf
that you are used to, except for the extended machine learning capabilities.
In the left toolbar, you can see an entry for Loud ML, and the Data Explorer
screen also includes the 1-Click ML button.

To stop all the containers:

```
$ docker-compose stop
```

To remove all Docker resources and all data that has been generated:

```
$ docker-compose rm
$ sudo rm -rf ./var
```

### Advanced usage

#### Loud ML

If you want to send direct commands to Loud ML, you have two options:

1. Use the REST API that is exposed on the port 8077.
2. Open a shell to run commands in the Loud ML container:

```
$ docker-compose exec loudml bash
```

#### Docker

The default settings are fine to test the TICK-L stack. If you need to make
a more advanced usage of Docker, there are several documents on the InfluxData
website that you could use as a guideline for more advanced settings.

## FAQ

### Error 'permission denied'

Running Docker with SELinux enabled will prevent the containers from
accessing the volumes. If you are in a lab environment, it may be
acceptable to disable SELinux by running a command such as:

```
$ setenforce permissive
$ vi /etc/selinux/config
```

In a production environment, you will need to configure SELinux as
required by your environment.

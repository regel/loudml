# Dockerfile for LoudML

This directory contains a `Dockerfile` that can be used with Docker Compose
to spawn a stack of containers with:
- InfluxDB
- LoudML Community Edition
- Chronograf for LoudML

The following ports are exposed on the host:
- 8086: InfluxDB
- 8077: LoudML
- 8888: Chronograf

The configuration file for LoudML `./etc/loudml.yml` is also required.

## Instructions

### Configuration

The default configuration creates two subdirectories in the current
directory:
- `./var/influxdb/`: InfluxDB database
- `./var/loudml/`: LoudML models

If you want to change those directoryies, you can edit the file
`docker-compose.yml` to change the lines below:

```yaml
services:
  loudml:
    volumes:
      - ./var/loudml:/var/lib/loudml
  influxdb:
    volumes:
      - ./var/influxdb:/var/lib/influxdb
```

### Basic usage

To start all the containers:

```
$ docker-compose up
```

You can open Chronograf on the host port 8888. This is the same Chronograf
that you are used to, except for the extended Machine-Learning capabilities.
In the left toolbar, you can see an entry for LoudML, and the Data Explorer
screen also includes the 1-Click ML button.

To stop all the containers:

```
$ docker-compose stop
```

### Advanced usage

If you want to send direct commands to LoudML, you have two options:

1. Use the REST API that is exposed on the port 8077.
2. Open a shell to run commands in the LoudML container:

```
$ docker-compose exec loudml bash
```

## FAQ

### Error 'permission denied'

Running Docker with SELinux enabled will prevent the containers from
accessing to the volumes. If you are in a lab environment, it may be
acceptable to disable SELinux by running a command such as:

```
$ setenforce permissive
$ vi /etc/selinux/config
```

In a production environment, you will have to configure SELinux as
required by your environment.

### I need Kapacitor

Make the following changes to the Docker Compose file:

```yaml
services:
  kapacitor:
    image: kapacitor:latest
    depends_on:
      - influxdb
    environment:
      KAPACITOR_INFLUXDB_0_URLS_0: http://influxdb:8086

  chronograf:
    [...]
    depends_on:
      [...]
      - kapacitor
    environment:
      [...]
      KAPACITOR_URL: http://kapacitor:9092
```

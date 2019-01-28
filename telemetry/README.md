# Telemetry

This package collects information about a running instance of Loud ML.

Because of regulation, personnal information cannot be stored in a database.
This applies to e-mail for instance.

## Requirements

InfluxDB listening on `localhost:8086` and with a `telemetry` database. The
database can be created with the command:

```
$ influx
> create database telemetry
```

## Deployment

Sample NGINX configuration file:

```
# /etc/nginx/default.d/telemetry.conf
location /api {
  proxy_pass http://127.0.0.1:8080/api;
}
```

## Data Visualization

Chronograf can be used to visualize the data from the InfluxDB database. Here
is an example of a query in InfluxQL language:

```sql
SELECT count(distinct("hostid")) AS "count_hostid"
FROM "telemetry"."autogen"."telemetry"
WHERE time > :dashboardTime: GROUP BY time(:interval:) FILL(previous)
```

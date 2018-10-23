#!/bin/bash
# Test licensing limitations.
#
# Requirements:
# - loudmld configuration
# - loudml, loudmld and loudml-lic must be in the PATH (eg. source venv)
# - openssl
#
# Optional:
# - InfluxDB
#
# The tests use the program 'loudml' to check license enforcement. The daemon
# 'loudmld' is not reloaded after a license update, but this is still
# acceptable because both the CLI and the daemon share the same code to load
# the license and create the models.
#
# WARNING: all models are removed as part of the testing process. Do not run
# this program in production.

set -e

config=""
keep_tmpdir=0
loudmld_pid=""
license=""
use_influx=0

# die MESSAGE
# Print message and exit with error code
die() {
    echo "$*" 1>&2
    exit 1
}

# finish
# Stop running processes and remove temporary files
finish() {
    loudmld_stop

    if [ $keep_tmpdir = 0 ]; then
        rm -rf "$tmpdir"
    fi
}

# usage
# Print usage screen
usage() {
    cat <<EOF
test_license.sh - Test Loud ML licensing limitations

Usage: test_license.sh OPTIONS

-i     use InfluxDB
-k     keep temporary directory
EOF
}

# loudmld_start
# Start Loud ML daemon and wait for it to be ready
loudmld_start() {
    echo "Starting Loud ML daemon..."
    loudmld -c "$config" &
    loudmld_pid=$!

    sleep 5
}

# loudmld_stop
# Stop Loud ML daemon
loudmld_stop() {
    if [ "$loudmld_pid" = "" ]; then
        return;
    fi

    echo "Stopping Loud ML daemon..."
    kill "$loudmld_pid"
    loudmld_pid=""
}

while getopts ":ik" option; do
    case "${option}" in
        i)
            use_influx=1
            ;;
        k)
            keep_tmpdir=1
            ;;
        *)
            usage
            die "Invalid argument"
            ;;
    esac
done
shift $((OPTIND-1))

if ! which loudml >/dev/null 2>&1; then
    die "Unable to run loudml executable"
fi

if ! which loudml-lic >/dev/null 2>&1; then
    die "Unable to run loudml executable"
fi

if ! which openssl >/dev/null 2>&1; then
    die "Unable to run openssl executable"
fi

tmpdir="$(mktemp -d)"
if [ $keep_tmpdir = 1 ]; then
    echo "Temporary directory: $tmpdir"
fi
trap finish EXIT

# Generate configuration file
config="$tmpdir/config.yml"
license="$tmpdir/license.lic"
if [ $use_influx = 1 ]; then
cat > "$config" <<EOF
---
datasources:
 - name: influx
   type: influxdb
   addr: localhost
   database: mydatabase

storage:
  path: $tmpdir/lib

server:
  listen: localhost:8077

license:
  path: $license
EOF
else
cat > "$config" <<EOF
---
datasources: []

storage:
  path: $tmpdir/lib

server:
  listen: localhost:8077

license:
  path: $license
EOF
fi

# Generate test keys
key_priv="$tmpdir/private.pem"
key_pub="$tmpdir/public.pem"
openssl genrsa -out "$key_priv" 2048
openssl rsa -in "$key_priv" -pubout -out "$key_pub"

# Create license data
data="$tmpdir/data.json"
cat >"$data" <<EOF
{
    "features": {
        "nrmodels": 1,
        "datasources": [ "elasticsearch", "influxdb" ],
        "models": [ "timeseries" ],
        "data_range": [ "2018-01-01", "2018-01-31" ]
    },
    "hostid": "any"
}
EOF

# Timeseries model
model1="$tmpdir/timeseries_model1.json"
model2="$tmpdir/timeseries_model2.json"
model3="$tmpdir/timeseries_model3.json"
model4="$tmpdir/fingerprints_model1.json"
cat >"$model1" <<EOF
{
    "bucket_interval": "1m",
    "default_datasource": "influx",
    "features": [
      {
        "default": 0,
        "field": "foo",
        "measurement": "temperature_series",
        "metric": "avg",
        "name": "avg_temp_feature"
      }
    ],
    "interval": 60,
    "max_evals": 10,
    "name": "avg_temp1-model",
    "offset": 30,
    "span": 5,
    "max_threshold": 90,
    "min_threshold": 50,
    "type": "timeseries"
}
EOF

cat >"$model4" <<EOF
{
  "timestamp_field": "@timestamp",
  "interval": "1m",
  "type": "fingerprints",
  "span": "7d",
  "key": "account_ref",
  "max_keys": 20000,
  "width": 50,
  "height": 50,
  "use_daytime": true,
  "daytime_interval": "6h",
  "offset": "30s",
  "aggregations": [
    {
      "measurement": "xdr",
      "features": [
        {
          "field": "duration",
          "name": "count-all",
          "metric": "count"
        },
        {
          "field": "duration",
          "name": "mean-all-duration",
          "metric": "avg"
        },
        {
          "field": "duration",
          "name": "std-all-duration",
          "metric": "stddev"
        }
      ]
    },
    {
      "measurement": "xdr",
      "match_all": [{"tag": "international", "value": true}],
      "features": [
        {
          "field": "duration",
          "name": "count-international",
          "metric": "count"
        },
        {
          "field": "duration",
          "name": "mean-international-duration",
          "metric": "avg"
        },
        {
          "field": "duration",
          "name": "std-international-duration",
          "metric": "stddev"
        }
      ]
    },
    {
      "measurement": "xdr",
      "match_all": [{"tag": "toll_call", "value": true}],
      "features": [
        {
          "field": "duration",
          "name": "count-premium",
          "metric": "count"
        },
        {
          "field": "duration",
          "name": "mean-premium-duration",
          "metric": "avg"
        },
        {
          "field": "duration",
          "name": "std-premium-duration",
          "metric": "stddev"
        }
      ]
    }
  ],
  "default_datasource": "elastic",
  "name": "fraud-model"
}

EOF

sed 's/avg_temp1-model/avg_temp2-model/' "$model1" > "$model2"
sed 's/avg_temp1-model/avg_temp3-model/' "$model1" > "$model3"

lic1="$tmpdir/lic1.lic"
loudml-lic -g "$lic1" --pub "$key_pub" --priv "$key_priv" --data "$data"

cp -f "$lic1" "$license"

loudmld_start

# Remove all models
echo "Removing all existing models..."
loudml -c "$config" list-models \
    | xargs -r -n 1 loudml -c "$config" delete-model

echo "Test: create model allowed by license"
loudml -c "$config" create-model "$model1"

echo "Test: model creation failure because of excessive number of models"
if loudml -c "$config" create-model "$model2"; then
    die "Expected failure"
fi

cat >"$data" <<EOF
{
    "features": {
        "nrmodels": 2,
        "datasources": [ "elasticsearch", "influxdb" ],
        "models": [ "timeseries" ]
    },
    "hostid": "any"
}
EOF

if [ $use_influx = 0 ]; then
    echo "Skippig test: training failure because data out of authorized date range"
else
    echo "Test: training failure because data out of authorized date range"
    if loudml -c "$config" train -d influx -f 2017-12-01 -t 2018-01-31 "avg_temp1-model"; then
        die "Expected failure"
    fi
fi

echo "New license"
loudml-lic -g "$license" --pub "$key_pub" --priv "$key_priv" --data "$data"

echo "Test: create model allowed by license"
loudml -c "$config" create-model "$model2";

cat >"$data" <<EOF
{
    "features": {
        "nrmodels": "unlimited",
        "datasources": [ "elasticsearch", "influxdb" ],
        "models": [ "timeseries" ]
    },
    "hostid": "any"
}
EOF

echo "New license"
loudml-lic -g "$license" --pub "$key_pub" --priv "$key_priv" --data "$data"

echo "Test: unlimited number of models"
loudml -c "$config" create-model "$model3";

echo "Test: unauthorized model type"
if loudml -c "$config" create-model "$model4"; then
    die "Expected failure"
fi

cat >"$data" <<EOF
{
    "features": {
        "nrmodels": "unlimited",
        "datasources": [ "elasticsearch", "influxdb" ],
        "models": [ "timeseries", "fingerprints" ]
    },
    "hostid": "any"
}
EOF

echo "New license"
loudml-lic -g "$license" --pub "$key_pub" --priv "$key_priv" --data "$data"

echo "Test: authorized model type"
loudml -c "$config" create-model "$model4"

loudmld_stop

echo "License test OK"

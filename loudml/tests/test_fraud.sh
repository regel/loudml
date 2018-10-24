#!/bin/bash
# Test fraud detection using CDR data
#
# Requirements:
# - loudmld configuration
# - loudml, loudmld and loudml-lic must be in the PATH (eg. source venv)
# - openssl
# - influxDB is running on the localhost
# - CDRs were imported into cdr.generic DB.measurement, and contains
# data from 01-02-2017 to 30-06-2017
#
# The tests use the program 'loudml' to check fingerprints CLI
# using a reference CDR data-set.
#

read -r -d '' test_ano_res << EOF
WARNING:root:detected anomaly for model '__name__' and key '1022320' at 2017-03-03T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1022320' at 2017-03-04T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1022320' at 2017-03-05T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1022320' at 2017-03-06T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1022320' at 2017-03-07T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1022320' at 2017-03-08T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1022320' at 2017-03-09T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1022320' at 2017-03-10T00:00:0000Z (score = 100)
INFO:root:anomaly ended for model '__name__' and key '1022320' at 2017-03-11T00:00:0000Z (score = 0)
WARNING:root:detected anomaly for model '__name__' and key '2022109' at 2017-05-14T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-05-15T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-05-16T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-05-17T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-05-18T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-05-19T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-05-20T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-05-21T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-05-22T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-05-23T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-05-24T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-05-25T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-05-26T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-05-27T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-05-28T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-05-29T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-05-30T00:00:0000Z (score = 100)
INFO:root:anomaly ended for model '__name__' and key '2022109' at 2017-05-31T00:00:0000Z (score = 43)
WARNING:root:detected anomaly for model '__name__' and key '1022320' at 2017-06-03T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1022320' at 2017-06-04T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1022320' at 2017-06-05T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1022320' at 2017-06-06T00:00:0000Z (score = 100)
INFO:root:anomaly ended for model '__name__' and key '1022320' at 2017-06-07T00:00:0000Z (score = 99)
WARNING:root:detected anomaly for model '__name__' and key '1013808' at 2017-06-14T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1013808' at 2017-06-15T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1013808' at 2017-06-16T00:00:0000Z (score = 100)
WARNING:root:detected anomaly for model '__name__' and key '2022109' at 2017-06-17T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1013808' at 2017-06-17T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-06-18T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1013808' at 2017-06-18T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-06-19T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1013808' at 2017-06-19T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-06-20T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1013808' at 2017-06-20T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-06-21T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1013808' at 2017-06-21T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-06-22T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '1013808' at 2017-06-22T00:00:0000Z (score = 100)
WARNING:root:anomaly still in progress for model '__name__' and key '2022109' at 2017-06-23T00:00:0000Z (score = 100)
INFO:root:anomaly ended for model '__name__' and key '1013808' at 2017-06-23T00:00:0000Z (score = 0)
INFO:root:anomaly ended for model '__name__' and key '2022109' at 2017-06-24T00:00:0000Z (score = 0)
EOF

set -e

MEASUREMENT=generic
TAGNAME=account
run_from="2017-02-01T00:00:00.000Z"
run_to="2017-07-01T00:00:00.000Z"
min_threshold="99"
max_threshold="99.9"
training_from="2017-03-01T00:00:00.000Z"
training_to="2017-04-01T00:00:00.000Z"

low_watermark=10
datasource=influx
database="cdr"
#dbaddr='localhost'
# Use remote reference database. Don't drop this database!
dbaddr='51.15.103.113:8086'
dbuser='5b970c'
dbuser_password='RgZ3i9wr270N'
config=""
keep_tmpdir=0
loudmld_pid=""
license=""

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
test_fraud.sh - Test Loud ML fingerprints CLI
Usage: test_fraud.sh OPTIONS
-k     keep temporary directory
EOF
}


# loudmld_start
# Start LoudML daemon and wait for it to be ready
loudmld_start() {
    echo "Starting LoudML daemon..."
    loudmld -c "$config" &
    loudmld_pid=$!

    sleep 5
}

# loudmld_stop
# Stop LoudML daemon
loudmld_stop() {
    if [ "$loudmld_pid" = "" ]; then
        return;
    fi

    echo "Stopping LoudML daemon..."
    kill "$loudmld_pid"
    loudmld_pid=""
}

while getopts ":k" option; do
    case "${option}" in
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
cat > "$config" <<EOF
---
datasources:
 - name: $datasource
   type: influxdb
   database: $database
   addr: $dbaddr
   dbuser: $dbuser
   dbuser_password: $dbuser_password
   database: $database
storage:
  path: $tmpdir/lib
server:
  listen: localhost:8077
license:
  path: $license
EOF


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
        "nrmodels": 20,
        "datasources": [ "elasticsearch", "influxdb" ],
        "models": [ "timeseries", "fingerprints" ]
    },
    "hostid": "any"
}
EOF


lic1="$tmpdir/lic1.lic"
loudml-lic -g "$lic1" --pub "$key_pub" --priv "$key_priv" --data "$data"
cp -f "$lic1" "$license"


loudmld_start


tmpfile=$(mktemp $tmpdir/test_fingerprint_XXXXXX)
name=$(basename $tmpfile)
cat << EOF > $tmpfile
{
    "aggregations": [
        {
            "features": [
                {
                    "anomaly_type": "high",
                    "field": "cost",
                    "metric": "sum",
                    "low_watermark": $low_watermark,
                    "name": "cost-0"
                }
            ],
            "match_all": [
                {
                    "tag": "direction",
                    "value": 0
                },
                {
                    "tag": "category",
                    "value": 0
                }
            ],
            "measurement": "$MEASUREMENT"
        },
        {
            "features": [
                {
                    "anomaly_type": "high",
                    "field": "cost",
                    "metric": "sum",
                    "low_watermark": $low_watermark,
                    "name": "cost-1"
                }
            ],
            "match_all": [
                {
                    "tag": "direction",
                    "value": 0
                },
                {
                    "tag": "category",
                    "value": 1
                }
            ],
            "measurement": "$MEASUREMENT"
        },
        {
            "features": [
                {
                    "anomaly_type": "high",
                    "field": "cost",
                    "metric": "sum",
                    "low_watermark": $low_watermark,
                    "name": "cost-2"
                }
            ],
            "match_all": [
                {
                    "tag": "direction",
                    "value": 0
                },
                {
                    "tag": "category",
                    "value": 2
                }
            ],
            "measurement": "$MEASUREMENT"
        },
        {
            "features": [
                {
                    "anomaly_type": "high",
                    "field": "cost",
                    "metric": "sum",
                    "low_watermark": $low_watermark,
                    "name": "cost-3"
                }
            ],
            "match_all": [
                {
                    "tag": "direction",
                    "value": 0
                },
                {
                    "tag": "category",
                    "value": 3
                }
            ],
            "measurement": "$MEASUREMENT"
        },
        {
            "features": [
                {
                    "anomaly_type": "high",
                    "field": "cost",
                    "metric": "sum",
                    "low_watermark": $low_watermark,
                    "name": "cost-4"
                }
            ],
            "match_all": [
                {
                    "tag": "direction",
                    "value": 0
                },
                {
                    "tag": "category",
                    "value": 4
                }
            ],
            "measurement": "$MEASUREMENT"
        },
        {
            "features": [
                {
                    "anomaly_type": "high",
                    "field": "cost",
                    "metric": "sum",
                    "low_watermark": $low_watermark,
                    "name": "cost-5"
                }
            ],
            "match_all": [
                {
                    "tag": "direction",
                    "value": 0
                },
                {
                    "tag": "category",
                    "value": 5
                }
            ],
            "measurement": "$MEASUREMENT"
        }
    ],
    "bucket_interval": "24h",
    "default_datasource": "$datasource",
    "height": 20,
    "width": 20,
    "interval": "1d",
    "key": "$TAGNAME",
    "max_keys": 20000,
    "max_threshold": 0,
    "min_threshold": 0,
    "offset": "30s",
    "span": "7d",
    "name": "$name",
    "type": "fingerprints"
}
EOF

echo "Test: creating model"
loudml -c "$config" create-model "$tmpfile"
echo "Test: training model in range $training_from - $training_to"
loudml -c "$config" train $name -f $training_from -t $training_to -l 1000
echo "Test: showing model stats"
loudml -c "$config" show-model $name -s

tmplog=$(mktemp $tmpdir/test_fingerprint_XXXXXX.log)
echo "Test: running anomaly detection in range $run_from - $run_to"
# change working dir, the files created by loudml run will be in $tmpdir
cd "$tmpdir"
loudml -c "$config" run $name -s -f $run_from -t $run_to -a -m $min_threshold -M $max_threshold 2> $tmplog

num_warn=$(grep -c "^WARNING" "$tmplog")
echo "Test: total anomalies raised"
if [ $num_warn -ne 977 ]
then
    die "# warnings = $num_warn whereas expected = 977"
fi

res=$(grep "anomaly" "$tmplog" | egrep "(1022320|2022109|1013808)" | sed 's/test_fingerprint_[a-zA-Z0-9]*//g;s/\.[0-9]//g' | sed 's/still in //' | sort -k 10 -k 8)
expected=$(echo "$test_ano_res" | sed 's/__name__//g;s/\.[0-9]//g' | sed 's/still in //' | sort -k 10 -k 8)
if [ "$res" != "$expected" ]
then
    echo "$res" > "$tmpdir/res"
    echo "$expected" > "$tmpdir/expected"
    diff -u "$tmpdir/expected" "$tmpdir/res"
    die "Not the expected anomalies"
fi

echo "Test: delete model"
loudml -c "$config" delete-model $name

loudmld_stop

echo "Fingerprint test OK"


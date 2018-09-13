#!/bin/bash
# Test fingerprints module.
#
# Requirements:
# - loudmld configuration
# - loudml, loudmld and loudml-lic must be in the PATH (eg. source venv)
# - openssl
# - influxDB is running on the localhost
#
# The tests use the program 'loudml' to check fingerprints CLI.
#

read -r -d '' fp_one << EOF
"fingerprint": [
1820.0
2037.0
2086.0
1946.0
1659.0
1302.0
973.0
749.0
707.0
840.0
1127.0
1484.0
]
EOF

read -r -d '' fp_two << EOF
"fingerprint": [
1820.0
2037.0
2086.0
1946.0
1659.0
1302.0
973.0
749.0
707.0
840.0
1127.0
1394.0
]
EOF

read -r -d '' log_one << EOF
WARNING:root:detected anomaly for model '__name__' and key 't1' at 2018-08-15T00:00:00.000Z (score = 99.4)
WARNING:root:anomaly still in progress for model '__name__' and key 't1' at 2018-08-16T00:00:00.000Z (score = 99.8)
WARNING:root:anomaly still in progress for model '__name__' and key 't1' at 2018-08-17T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't1' at 2018-08-18T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't1' at 2018-08-19T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't1' at 2018-08-20T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't1' at 2018-08-21T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't1' at 2018-08-22T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't1' at 2018-08-23T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't1' at 2018-08-24T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't1' at 2018-08-25T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't1' at 2018-08-26T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't1' at 2018-08-27T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't1' at 2018-08-28T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't1' at 2018-08-29T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't1' at 2018-08-30T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't1' at 2018-08-31T00:00:00.000Z (score = 100.0)
EOF

read -r -d '' log_two << EOF
WARNING:root:detected anomaly for model '__name__' and key 't2' at 2018-08-12T00:00:00.000Z (score = 99.1)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-13T00:00:00.000Z (score = 99.9)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-14T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-15T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-16T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-17T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-18T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-19T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-20T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-21T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-22T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-23T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-24T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-25T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-26T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-27T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-28T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-29T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-30T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't2' at 2018-08-31T00:00:00.000Z (score = 100.0)
EOF

read -r -d '' log_three << EOF
WARNING:root:detected anomaly for model '__name__' and key 't3' at 2018-08-11T00:00:00.000Z (score = 99.4)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-12T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-13T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-14T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-15T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-16T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-17T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-18T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-19T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-20T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-21T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-22T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-23T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-24T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-25T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-26T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-27T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-28T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-29T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-30T00:00:00.000Z (score = 100.0)
WARNING:root:anomaly still in progress for model '__name__' and key 't3' at 2018-08-31T00:00:00.000Z (score = 100.0)
EOF

set -e

MEASUREMENT=me
TAGNAME=tag
FROM="2018-08-01T00:00:00.000Z"
TO="2018-09-01T00:00:00.000Z"
STEP=$((3600 * 1000))

datasource=influx
database="test-$(date +%s)"
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
    rm -rf "$tmpdir"
    influx -database $database -execute "DROP DATABASE \"$database\""
}

# usage
# Print usage screen
usage() {
    cat <<EOF
test_fingerprints.sh - Test Loud ML fingerprints CLI
Usage: test_fingerprints.sh OPTIONS
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
   addr: localhost
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


function get_extra_opts() {
    from=$1
    to=$2
    sigma=$3
    shape=$4
    period=$5
    max_val=$6
    min_val=$7
    trend=$8
    echo "--shape $shape --from $from --to $to --period $period --sigma $sigma --step-ms $STEP --base $min_val --amplitude $max_val --trend $trend"
}

function gen_times_data() {
    datasource=$1
    measurement=$2
    tags=$3
    opts="$4"
    loudml-faker -c "$config" -o $datasource -m $measurement --tags $tags $opts 
}

tags=("si1" "si2" "si3"  "sa1" "sa2" "sa3" "f1" "f2" "f3" "t1" "t2" "t3" )
shapes=("sin" "sin" "sin" "saw" "saw" "saw" "flat" "flat" "flat" "triangle" "triangle" "triangle")
sigmas=("0" "4" "8" "2" "4" "8"  "2" "4" "8" "0" "0" "0")
periods=("$((24 * 3600))" \
         "$((24 * 3600))" \
         "$((24 * 3600))" \
         "$((24 * 3600))" \
         "$((24 * 3600))" \
         "$((24 * 3600))" \
         "$((24 * 3600))" \
         "$((24 * 3600))" \
         "$((24 * 3600))" \
         "$((24 * 3600))" \
         "$((24 * 3600))" \
         "$((24 * 3600))")

min_vals=(100 150 200 \
          100 150 200 \
          100 150 200 \
          100 150 200)
max_vals=(50 60 70 \
          50 60 70 \
          50 60 70 \
          50 60 70)
trends=(0 0 0
        0 0 0
        0 0 0
        3 5 7)

loudmld_start

echo "Test: generating test data"
i=0
for tag in "${tags[@]}"
do
    from=$FROM
    to=$TO
    shape=${shapes[$i]} 
    sigma=${sigmas[$i]}
    period=${periods[$i]}
    max_val=${max_vals[$i]}
    min_val=${min_vals[$i]}
    trend=${trends[$i]}

    options=$(get_extra_opts $from $to $sigma $shape $period $max_val $min_val $trend)
    gen_times_data $datasource $MEASUREMENT "$TAGNAME:$tag" "$options"
    i=$(($i + 1))
done

tmpfile=$(mktemp $tmpdir/test_fingerprint_XXXXXX)
name=$(basename $tmpfile)
cat << EOF > $tmpfile
{
    "aggregations": [
        {
            "features": [
                {
                    "anomaly_type": "low_high",
                    "field": "value",
                    "metric": "count",
                    "name": "count_val"
                }
            ],
            "measurement": "$MEASUREMENT"
        }
    ],
    "daytime_interval": "2h",
    "default_datasource": "$datasource",
    "height": 5,
    "width": 5,
    "interval": "1d",
    "key": "$TAGNAME",
    "max_keys": 200,
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
from=$(date +%s --utc --date="$FROM")
to=$(($from + 7*24*3600))
echo "Test: training model in range $from - $to"
loudml -c "$config" train $name -f $from -t $to -l 1000
echo "Test: showing model stats"
loudml -c "$config" show-model $name -s


from=$(date +%s --utc --date="$FROM")
to=$(($from + 7*24*3600))
echo "Test: fingerprint value for tag si1 in range $from - $to"
res=$(loudml -c "$config" predict $name -f $from -t $to -k "si1" | awk '/"fingerprint":/,/]/' | sed 's/^ *//;s/,//')
expected="$fp_one"
if [ "$res" != "$expected" ]
then
    echo "$res" #| hexdump -vC
    die "bad fingerprint"
fi



to=$(date +%s --utc --date="$TO")
from=$(($to - 7*24*3600))
echo "Test: fingerprint value for tag si1 in range $from - $to"
res=$(loudml -c "$config" predict $name -f $from -t $to -k "si1" | awk '/"fingerprint":/,/]/' | sed 's/^ *//;s/,//')
expected="$fp_two"
if [ "$res" != "$expected" ]
then
    echo "$res" #| hexdump -vC
    die "bad fingerprint"
fi


tmplog=$(mktemp $tmpdir/test_fingerprint_XXXXXX.log)
# t1, t2, and t3 have an increasing trend == their fingerprint changes over time
# and triggers 'high' anomalies
echo "Test: running anomaly detection in range $FROM - $TO"
loudml -c "$config" run $name -f $FROM -t $TO -a -m 95 -M 99 2> $tmplog

num_warn=$(grep -c "^WARNING" $tmplog)
echo "Test: total anomalies raised"
if [ $num_warn -ne 58 ]
then
    die "# warnings = $num_warn whereas expected = 58"
fi


echo "Test: anomalies for tag t1"
res=$(grep "^WARNING" $tmplog | grep "t1" | sed 's/\.[0-9]//g')
expected=$(echo "$log_one" | sed "s/__name__/$name/g" | sed 's/\.[0-9]//g')
if [ "$res" != "$expected" ]
then
    echo "$res" #| hexdump -vC
    die "Not the expected anomalies for tag t1"
fi

echo "Test: anomalies for tag t2"
res=$(grep "^WARNING" $tmplog | grep "t2" | sed 's/\.[0-9]//g')
expected=$(echo "$log_two" | sed "s/__name__/$name/g" | sed 's/\.[0-9]//g')
if [ "$res" != "$expected" ]
then
    echo "$res" #| hexdump -vC
    die "Not the expected anomalies for tag t2"
fi

echo "Test: anomalies for tag t3"
res=$(grep "^WARNING" $tmplog | grep "t3" | sed 's/\.[0-9]//g')
expected=$(echo "$log_three" | sed "s/__name__/$name/g" | sed 's/\.[0-9]//g')
if [ "$res" != "$expected" ]
then
    echo "$res" #| hexdump -vC
    die "Not the expected anomalies for tag t3"
fi

echo "Test: delete model"
loudml -c "$config" delete-model $name

loudmld_stop

echo "Fingerprint test OK"


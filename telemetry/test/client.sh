#!/bin/bash
# Connect to telemetry server

# ---------------------------------------------------------------------------
# Configuration

# Telemetry server
host="localhost:8080"

# End of configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test: invalid method
json='{"host_id":"myhostid", "loudml":{"version": "1.2.3"}}'

args=()
args+=(-X "GET")
args+=(-H "Content-Type: application/json")
args+=(-d "$json")
args+=("http://$host/api")

echo "Test: invalid method"
curl "${args[@]}"

# ---------------------------------------------------------------------------
# Test: invalid json
json='invalid_json'

args=()
args+=(-X "POST")
args+=(-H "Content-Type: application/json")
args+=(-d "$json")
args+=("http://$host/api")

echo "Test: invalid JSON"
curl "${args[@]}"

# ---------------------------------------------------------------------------
# Test: required fields missing
json='{}'

args=()
args+=(-X "POST")
args+=(-H "Content-Type: application/json")
args+=(-d "$json")
args+=("http://$host/api")

echo "Test: required fields missing"
curl "${args[@]}"

# ---------------------------------------------------------------------------
# Test: JSON OK
json='{"host_id":"myhostid", "loudml":{"version": "1.2.3"}}'

args=()
args+=(-X "POST")
args+=(-H "Content-Type: application/json")
args+=(-d "$json")
args+=("http://$host/api")

echo "Test: JSON OK"
curl "${args[@]}"

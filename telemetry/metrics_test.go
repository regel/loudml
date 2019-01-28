package main

import "reflect"
import "testing"

// Test various cases of JSON decoding
func TestMetricsFromJSON(t *testing.T) {
	var m Metrics
	var err error
	json1 := []byte(`{invalid json`)
	json2 := []byte(`{"key": "value"}`)
	json3 := []byte(`{"host_id": "dead-beef", "loudml": {"version": "1.2.3"}}`)

	m, err = metricsFromJSON(json1)
	if err == nil {
		t.Errorf("error: invalid JSON expected")
	}

	m, err = metricsFromJSON(json2)
	if err == nil {
		t.Errorf("error: expected missing fields in JSON")
	}

	m, err = metricsFromJSON(json3)
	if m.HostId != "dead-beef" {
		t.Errorf("error: expected 'dead-beef', got '%s'", m.HostId)
	}
}

// Test serialization of Metrics structure using InfluxDB Line protocol
func TestMetricsToInfluxDB(t *testing.T) {
	m := Metrics{HostId: "myhostid"}
	m.LoudML.Version = "1.2.3"
	expected := "telemetry,version=1.2.3 hostid=myhostid 12345"
	var res string

	res = m.toInfluxDBLine("12345", "telemetry")
	if res != expected {
		t.Errorf("error: expected '%s', got '%s'", expected, res)
	}
}

func TestMetricTags(t *testing.T) {
	m := Metrics{HostId: "myhostid"}
	m.LoudML.Version = "1.2.3"
	expect := map[string]string{
		"version": "1.2.3",
	}
	res := m.tags()
	if !reflect.DeepEqual(res, expect) {
		t.Errorf("error: bad tags map")
	}
}

func TestMetricFields(t *testing.T) {
	m := Metrics{HostId: "myhostid"}
	m.LoudML.Version = "1.2.3"
	expected := map[string]interface{}{
		"hostid": m.HostId,
	}
	res := m.fields()
	if !reflect.DeepEqual(res, expected) {
		t.Errorf("error: expected '%s', got '%s'", expected, res)
	}
}

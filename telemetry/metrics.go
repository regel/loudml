package main

import (
	"encoding/json"
	"errors"
)

type Metrics struct {
	HostId string `json:"host_id"`
	LoudML struct {
		Version string `json:"version"`
	} `json:"loudml"`
}

func metricsFromJSON(jsonBlob []byte) (Metrics, error) {
	var m Metrics

	if !json.Valid(jsonBlob) {
		return m, errors.New("invalid JSON")
	}

	err := json.Unmarshal(jsonBlob, &m)
	if err != nil {
		return m, err
	}

	if m.HostId == "" {
		return m, errors.New("missing field host_id")
	}
	if m.LoudML.Version == "" {
		return m, errors.New("missing field loudml.version")
	}

	return m, nil
}

func (m Metrics) toInfluxDBLine(ts string, measurement string) string {
	return measurement + "," + "version=" + m.LoudML.Version + " " +
		"hostid=" + m.HostId + " " + ts
}

func (m Metrics) fields() map[string]interface{} {
	res := map[string]interface{}{
		"hostid": m.HostId,
	}

	return res
}

func (m Metrics) tags() map[string]string {
	res := map[string]string{
		"version": m.LoudML.Version,
	}

	return res
}

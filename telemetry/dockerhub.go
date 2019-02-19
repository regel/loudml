package main

import (
	"encoding/json"
	"errors"
)

type ImageInfo struct {
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
	PullCount int    `json:"pull_count"`
	StarCount int    `json:"star_count"`
	// other fields ignored
}

func imageInfoFromJSON(jsonBlob []byte) (ImageInfo, error) {
	i := ImageInfo{StarCount: -1, PullCount: -1}

	if !json.Valid(jsonBlob) {
		return i, errors.New("invalid JSON")
	}

	err := json.Unmarshal(jsonBlob, &i)
	if err != nil {
		return i, err
	}

	if i.Name == "" {
		return i, errors.New("missing field name")
	}

	if i.Namespace == "" {
		return i, errors.New("missing field namespace")
	}

	if i.PullCount == -1 {
		return i, errors.New("missing field pull_count")
	}

	if i.StarCount == -1 {
		return i, errors.New("missing field start_count")
	}

	return i, nil
}

func (i ImageInfo) fields() map[string]interface{} {
	res := map[string]interface{}{
		"pull_count": i.PullCount,
		"star_count": i.StarCount,
	}

	return res
}

func (i ImageInfo) tags() map[string]string {
	res := map[string]string{
		"user-agent": "docker",
		"image":      i.Namespace + "/" + i.Name,
	}

	return res
}

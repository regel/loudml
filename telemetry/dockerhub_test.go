package main

import (
	"reflect"
	"testing"
)

func TestDockerHubParse(t *testing.T) {
	var i ImageInfo
	json1 := []byte(`
{"user": "loudml", "name": "community", "namespace": "loudml", "repository_type": "image", "status": 1, "description": "Loud ML is the first open-source AI solution for ICT and IoT automation.", "is_private": false, "is_automated": false, "can_edit": false, "star_count": 3, "pull_count": 28844, "last_updated": "2019-02-16T06:44:05.637411Z", "is_migrated": false, "has_starred": false, "full_description": "...", "affiliation": null, "permissions": {"read": true, "write": false, "admin": false}}
`)

	i, _ = imageInfoFromJSON(json1)
	if i.StarCount != 3 {
		t.Errorf("error: expected %d, got %d", 3, i.StarCount)
	}
	if i.PullCount != 28844 {
		t.Errorf("error: expected %d, got %d", 28844, i.PullCount)
	}
	if i.Name != "community" {
		t.Errorf("error: expected '%s', got '%s'", "community", i.Name)
	}
	if i.Namespace != "loudml" {
		t.Errorf("error: expected '%s', got '%s'", "loudml", i.Name)
	}
}

func NewImageInfo() ImageInfo {
	i := ImageInfo{
		Name:      "community",
		Namespace: "loudml",
		PullCount: 10000,
		StarCount: 3,
	}

	return i
}

func TestImageInfoTags(t *testing.T) {
	i := NewImageInfo()
	expect := map[string]string{
		"user-agent": "docker",
		"image":      "loudml/community",
	}
	res := i.tags()
	if !reflect.DeepEqual(res, expect) {
		t.Errorf("error: expected '%s', got '%s'", expect, res)
	}
}

func TestImageInfoFields(t *testing.T) {
	i := NewImageInfo()
	expect := map[string]interface{}{
		"pull_count": i.PullCount,
		"star_count": i.StarCount,
	}
	res := i.fields()
	if !reflect.DeepEqual(res, expect) {
		t.Errorf("error: expected '%s', got '%s'", expect, res)
	}
}

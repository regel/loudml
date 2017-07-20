# Native Script Plugin for Elasticsearch

##. Important Deprecation Note

Please note that native scripts were [deprecated in v5.5.0](https://github.com/elastic/elasticsearch/pull/24692) and [removed in v6.0.0](https://github.com/elastic/elasticsearch/pull/24726). Consider migrating your native scripts to the ScriptEngine. Please see [elastic/elasticsearch#24726](https://github.com/elastic/elasticsearch/pull/24726) for more information. 


## Introduction

This plugin contains [native script](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-scripting-native.html) extensions for Elasticsearch.

Please make sure to use the correct branch of this repository that corresponds to the version of elasticsearch that you are developing the plugin for.

### Usage

The native script named **UserAgentSearch** will match params.field value against internal regular expressions and return **true** or **false**:
 * if params.device is not null, the script will return true for documents matching the given type
 * if params.model is not null, the script will return true for documents matching the given model
 * if params.browser is not null, the script will return true for documents matching the given browser

The above script can be used in search context, to refine queries, e.g.:

```bash
curl -X GET localhost:9200/beat*/_search -d'
{
  "query": {
    "bool": {
      "must": [
        {
          "script": {
            "script": {
              "inline": "UserAgentSearch", 
              "params": {
                "device": "smartphone",
                "field": "UserAgent"
              },
              "lang": "native"
            }
          }
        }
      ]
    }
  }
}'
```

The same native script can be used in aggregation or script field context and return the raw value directly.
 * if params.operation equals "getdevice", the script will return the device type value
 * if params.operation equals "getmodel", the script will return the device model value
 * if params.operation equals "getbrowser", the script will return the browser value

```bash
curl -X GET localhost:9200/beat*/_search -d'
{
  "size": 0,
  "timeout": "1s",
  "aggs": {
    "devices": {
      "terms": {
        "script": {
          "inline": "UserAgentSearch",
          "lang": "native",
          "params": {
            "operation": "getdevice",
            "field": "UserAgent"
          }
        }
      }
    }
  }
}'
```

## Building

```bash
export JAVA_HOME=/usr/lib/jvm/<your jvm>/
export PATH=$JAVA_HOME/bin:$PATH
./gradlew assemble
```
The assembled plugin can be found in the **build/distributions** directory. Follow the [elasticsearch instruction](https://www.elastic.co/guide/en/elasticsearch/plugins/current/plugin-management-custom-url.html) to install the plugin.

## Install and Upgrade

```bash
/usr/share/elasticsearch/bin/elasticsearch-plugin remove beat-native-script
-> removing [beat-native-script]...
/usr/share/elasticsearch/bin/elasticsearch-plugin install file://$PWD/java/native/build/distributions/beat-native-script-5.4.1.0.zip
-> Downloading file:///tmp/java/native/build/distributions/beat-native-script-5.4.1.0.zip
[=================================================] 100%   
-> Installed beat-native-script
# systemctl restart elasticsearch
```


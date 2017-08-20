# bonsai

A REST API to run ML jobs using Tensorflow

# Adding a model

A time series model can be defined and saved in the configuration,
yet remain inactive (stopped).
This operation uses the create API and requires query parameters:
 * name: A unique model name
 * index: The index pattern, to query data
 * interval: The interval (in seconds) to run periodic predict operations
 * bucket_interval: The interval (in seconds) that defines bucket width in date histogram aggregation
 * span: The interval (in seconds) that is relevant when predicting future values

The query body must define the features used to create the model
Each feature defines:
 - name: A unique feature name
 - metric: The metric used in aggregation, can be: count, min, max, sum, avg, variance, std_deviation
 - field: A field name present in ES documents
 - script: A Painless script, if field parameter is not used

```bash
curl -X POST -H 'Content-Type: application/json' "localhost:8077/api/model/create?name=foo&index=voip-*&interval=60&span=3600&bucket_interval=1200" -d '
{
"features": [
    {"name": "avg_duration", "metric": "avg", "script": "def val = doc['"'"'end_date'"'"'].value; if(val != 0) return (val - doc['"'"'@timestamp'"'"'].value)"}
 ]
}
'
```

# List existing models

FIXME: Write documentation

# Deleting a model

FIXME: Write documentation

# Training a model

FIXME: Write documentation

# Predicting values and detecting anomalies

FIXME: Write documentation

# Running the core App

NOTE: the model must be defined in the configuration. See 'Adding a model'

Training a model and saving model weights in the configuration:

```bash
START=$(( ($(date +%s)) - 3600*24*10 )) 
END=$(( ($(date +%s)) - 3600*24*6 )) 
python3 -m bonsai.compute -t --model foo -s $START -e $END 127.0.0.1:9200 
```

Predicting next values in real time mode:

```bash
python3 -m bonsai.compute -r -p --model foo 127.0.0.1:9200 
```



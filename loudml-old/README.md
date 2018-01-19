# loudml

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
python3 -m loudml.times -t --model foo -s $START -e $END 127.0.0.1:9200 
```

Predicting next values in real time mode:

```bash
python3 -m loudml.times -r -p --model foo 127.0.0.1:9200 
```

# Unsupervised Anomaly Detection

First things first, create the model. Here we define a model
 - that will fetch data in 'bananas*' indexes,
 - that will partition the data set into time series using the 'banana_ref' term(key),
 - that will run anomaly detection at periodic time 'interval'
 - that will aggregate data over $span time range (the short term signature) for the purpose
of anomaly detection

```bash
# short term signatures with 7 days history (the default)
span=$((7*24*3600))
curl -s -X POST "localhost:8077/api/ivoip/create?name=$name&index=bananas*&term=banana_ref&max_terms=20000&span=$span&interval=3600"
```

Once defined, run training. The from_date and to_date define the time range
that is used to compute the long term signatures.
The default range is 30 days (the last 30 days compared to current time)


```bash
python3 -m loudml.ivoip -m <name> --train -s $(date +%s --date="2017/01/01") -e $(date +%s --date="2017/01/31") 

Or (equivalent) using the REST API:
from=$(date +%s --date="2017/01/01")
to=$(date +%s --date="2017/01/31")
curl -s -X POST localhost:8077/api/ivoip/train?name=$name&from_date=$from&to_date=$to

The REST API returns a job_id. Training is asynchronous and runs in the background.
When training is complete the result is saved in .loudml index in Elasticsearch so that
it can be reloaded later.
```

To train the model faster, you can limit the number of profiles used in training.
The -l (Or limit) parameter will tell the framework to shuffle and select random
profiles for calculating clusters.
This will greatly reduce training time and should not affect the quality of the
clustering.

```bash
python3 -m loudml.ivoip -m <name> --train -s $(date +%s --date="2017/01/01") -e $(date +%s --date="2017/01/31") -l 1000

Or using the REST API:
from=$(date +%s --date="2017/01/01")
to=$(date +%s --date="2017/01/31")
curl -s -X POST localhost:8077/api/ivoip/train?name=$name&from_date=$from&to_date=$to&limit=1000
```

To replay data with the fraud detection model, run:

```bash
python3 -m loudml.ivoip -m <name> -p -R -s $(date +%s --date="2017/03/04") -e $(date +%s --date="2017/03/15") --threshold 99
```

For a quick scan (interval=24 hours), use the -i option:

```bash
curl -s -X DELETE localhost:9200/.loudml-anomalies
python3 -m loudml.ivoip -m <name> -p -R -s $(date +%s --date="2016/12/01") -e $(date +%s --date="2017/03/31") --threshold 99 -i $((24 * 3600))
curl -s localhost:9200/.loudml-anomalies/_search?size=1000
```




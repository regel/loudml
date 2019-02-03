### Input Data Source

This section is for developers who want to connect Loud ML to new
types of databases and enhance the compatibility of the software with
ICT data stores.

This interface allows for users to quickly connect their data to 
state of the art capabilities and focus on the core business and
operations.

Input data source authorship is kept as simple as possible to
promote people to develop and submit new inputs.

### Input Data Source Guidelines

- A new data source must conform to the [Datasource][] interface
- A new data source class should call `super.__init__()` in their `__init__` function to register
  themselves and inherit from the Datasource class
- A new data source class must be defined in a new file
- Declare settings in a voluptuous schema to validate their format
- The data source must be able to use the settings defined in `config.yml` file
- Methods that are not implemented must raise `raise NotImplemented()` exception
- The main `Description` header should reference relevant documentation if any
- Unit tests must load and save test data in the format defined for other sources
- Follow the recommended [CodeStyle][]

Let's say you've written a new data source for SQL data.

### SQL Example

```python

# sql.py
from loudml.datasource import DataSource
class MyDataSource(DataSource):
    """
    My datasource
    """

    SCHEMA = DataSource.SCHEMA.extend({
        Required('string_param'): str,
        Optional('bool_param', default=True): Boolean(),
        Optional('key_param'): All(schemas.key, Length(max=256)),
    })

    def __init__(self, cfg):
        cfg['type'] = 'mysource'
        super().__init__(cfg)

    def get_times_data(
        self,
        model,
        from_date=None,
        to_date=None,
    ):
        features = model.features
        nb_features = len(features)

        queries = self._build_times_queries(model, from_date, to_date)
        results = self._run_queries(queries)

        if not isinstance(results, list):
            results = [results]

        buckets = []
        # Merge results
        for i, result in enumerate(results):
            feature = features[i]

            for j, point in enumerate(result.get_points()):
                agg_val = point.get(feature.name)
                timeval = point['time']

                if j < len(buckets):
                    bucket = buckets[j]
                else:
                    bucket = {
                        'time': timeval,
                        'mod': int(str_to_ts(timeval)) % model.bucket_interval,
                        'values': {},
                    }
                    buckets.append(bucket)

                bucket['values'][feature.name] = agg_val

        # Build final result
        t0 = None
        result = []

        for bucket in buckets:
            X = np.full(nb_features, np.nan, dtype=float)
            timeval = bucket['time']
            ts = str_to_ts(timeval)

            for i, feature in enumerate(features):
                agg_val = bucket['values'].get(feature.name)
                if agg_val is None:
                    logging.info(
                        "missing data: field '%s', metric '%s', bucket: %s",
                        feature.field, feature.metric, timeval,
                    )
                else:
                    X[i] = agg_val

            if t0 is None:
                t0 = ts

            result.append(((ts - t0) / 1000, X, timeval))

        return result
```

### Development

* Add new Python code to the `loudml/loudml` directory
* Add new Python file test_source.py to the `loudml/tests` directory
* Run `make rpm` to package the new version

[CodeStyle]: https://github.com/regel/loudml/wiki/CodeStyle
[Datasource]: https://updateurl

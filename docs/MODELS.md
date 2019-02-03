### Machine Learning Model

This section is for developers and data scientists who want to
create new model types for time series and other data changing over
time.

This interface allows for users to pick and chose what is best for their
use cases and makes it easy for developers to create new ways of
deriving value from their data.

Model authorship follows the following guidelines to deliver consistent
and high quality implementations.

### Model Guidelines

- A new model must conform to the [Model][] interface
- A new model class should call `super.__init__()` in their `__init__` function to register
  themselves and inherit from the Model class
- A new model class must be defined in a new file
- The model must be able to serialize and deserialize the training state in JSON format
- Methods that are not implemented must raise `raise NotImplemented()` an exception
- The main `Description` header must reference the [ArXiv](https://arxiv.org)
  publication that motivates the new model implementation
- Unit tests include both training and inference
- Follow the recommended [CodeStyle][]

Let's say you've written a new model that fits a gaussian to input data.

### Gaussian Model Example

```python
"""My new Gaussian model

# Reference:
- [ArXiv article](
    https://arxiv.org/xxxxx)
"""

# gaussian.py

class GaussianModel(Model):
    """
    Time-series Gaussian model
    """
    TYPE = 'gaussian'

    SCHEMA = Model.SCHEMA.extend({
        Optional('new_param'): Any(None, "auto", All(int, Range(min=1))),
    })

    def __init__(self, settings, state=None):
        super().__init__(settings, state)

        settings = self.validate(settings)

    @property
    def type(self):
        return self.TYPE

    def train(
        self,
        datasource,
        from_date,
        to_date="now",
        train_size=0.67,
        batch_size=256,
        num_epochs=100,
        num_cpus=1,
        num_gpus=0,
        max_evals=None,
        progress_cb=None,
        license=None,
        incremental=False,
        windows=[],
    ):
        """
        Train model
        """
        return {
            "loss": loss_value,
        }
```

### Development

* Add new Python code to the `loudml/loudml` directory
* Add new Python file test_model.py to the `loudml/tests` directory
* Run `make rpm` to package the new version

### Data Formats

Your model can query data using a [source] interface and a time range.
Returned data can be stored into a NumPy array or other structures you
need in your model implementation.

[CodeStyle]: https://github.com/regel/loudml/wiki/CodeStyle
[Model]: https://updateurl
[source]: https://updateurl

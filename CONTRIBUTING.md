### Contributing

1. Open a [new issue][] to discuss the changes you would like to make.  This is
   not strictly required but it may help reduce the amount of rework you need
   to do later.
2. Make changes or write your code using the guidelines in the following
   documents:
   - [New data source type][datasources]
   - [New model type][models]
3. Ensure you have added proper unit tests and documentation.
4. Open a new [pull request][].

### Coding Style

Python code should be compliant with PEP8. Make sure that the linting tool
(flake8 or other) is running with Python 3, otherwise the tool may report
false positives due to changes between Python 2 and Python 3.

### PyDoc

Public interfaces for new data sources, models, and the model server,
can be found in the Python doc.

[![PyDoc](https://updateurl)](https://updateurl)

### Common development tasks

**Adding a dependency:**

Assuming you can already package and run the project, add the new dependency
in base/vendor/requirements.txt.in file, and run make before you commit
the new file:

1. `cd base; make vendor/requirements.txt`

**Unit Tests:**

Before opening a pull request you should run the short tests.

**Run short tests:**

```
nosetests-3.4 loudml/tests
```

**Execute integration tests:**

Running the integration tests requires several docker containers to be
running.  You can start the containers with:
```
make docker-run
```

And run the full test suite with:
```
make test-all
```

Use `make docker-kill` to stop the containers.


[new issue]: https://github.com/regel/loudml/issues/new/choose
[pull request]: https://github.com/regel/loudml/compare
[models]: /docs/MODELS.md
[datasources]: /docs/DATASOURCES.md

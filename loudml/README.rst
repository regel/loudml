Python Dependencies
===================

Python dependencies may be tricky to manage when it comes to delivering
software. In all cases, the dependencies must be added to the file
``setup.py``.

Dependencies Provided by RedHat/EPEL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dependency must be added to the SPEC file.

```
Requires: python34-numpy
```

Dependencies Provided by Debian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Debian helper tools parse the source code to autodetect the dependencies
that are already packaged. It uses an hardcoded list of import and packages
that is maintained upstream. As a consequence, it should not be necessary to
explicit the dependencies.

Dependencies Not Provided by the OS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section applies to dependencies that are not packaged by RedHat/EPEL,
Debian, or none of them.

To ship all those dependencies, the program uses a *vendor* system. Basically
we use the Python module path ``sys.path`` to load modules from a custom
directory. To do this, first the modules must be added to the package
``loudml-base`` (see directory ``base`` for details). Then, every program
entry point must import the module that updates the import path.

```
import loudml.vendor
```

Because this instruction has an impact on how the modules are loaded, it
should be one of the first modules that are imported.

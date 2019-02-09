import contextlib
from io import StringIO
import os
import tempfile
import unittest

import loudml.cli

CONFIG = """
---
datasources:
  - name: influx
    type: influxdb
    addr: localhost
    database: dummy_db

  - name: elastic
    type: elasticsearch
    addr: localhost:9200
    index: dummy-idx

storage:
  path: {}

server:
  listen: localhost:8077
"""


def execute(cmd):
    out = StringIO()

    with contextlib.redirect_stdout(out):
        try:
            loudml.cli.main(cmd)
        except Exception as exn:
            print(out.getvalue())
            raise exn

    return out.getvalue().strip()


class TestCli(unittest.TestCase):
    def setUp(self):
        # Storage directory
        self.tmp = tempfile.TemporaryDirectory()

        # Configuration
        _, self.config = tempfile.mkstemp(suffix='.yml')
        with open(self.config, 'w') as cfg_file:
            cfg_file.write(CONFIG.format(self.tmp.name))

        # Model example
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = os.path.join(
            script_dir,
            "..",
            "examples",
            "model-timeseries-influx.yml",
        )

    def tearDown(self):
        os.unlink(self.config)

    def test_commands(self):
        commands = set([name for name, _ in loudml.cli.get_commands()])

        self.assertEqual(
            commands,
            set([
                "create-model",
                "delete-model",
                "list-models",
                "predict",
                "forecast",
                "train",
                "show-model",
                "load-data",
                "plot",
                "load-checkpoint",
                "save-checkpoint",
                "list-checkpoints",
                ]),
        )

    def test_create_list_delete(self):
        base = ['-c', self.config]

        # Create
        execute(base + ['create-model', self.model])

        # List
        out = execute(base + ['list-models'])
        self.assertEqual(out, "timeseries-influx")

        # Delete
        execute(base + ['delete-model', 'timeseries-influx'])
        out = execute(base + ['list-models'])
        self.assertEqual(out, "")

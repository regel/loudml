import unittest

import loudml.vendor
from loudml import (
    errors,
    schemas,
)

class TestSchemas(unittest.TestCase):
    def valid(self, value):
        schemas.validate(self.schema, value)

    def invalid(self, value):
        with self.assertRaises(errors.Invalid):
            schemas.validate(self.schema, value)

    def test_key(self):
        self.schema = schemas.key

        self.valid("foo")
        self.valid("foo_bar")
        self.valid("Foo-Bar")
        self.valid("00_foo_00_bar_001")
        self.valid("_foo")

        self.invalid("")
        self.invalid("foo/bar")
        self.invalid(".foo")
